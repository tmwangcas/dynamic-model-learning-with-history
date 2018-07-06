import gym.spaces
import numpy as np
import tensorflow as tf
import os
import time
from dynamic_model import DynamicModel

# start time
start = time.time()

##################################################
################# Initialization #################
##################################################

print("\n##################################################")
print("Initializing settings")
print("##################################################\n")

# make directories for saving data
save_dir = 'saved_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir + '/training_data')
    os.makedirs(save_dir + '/validation_data')
    os.makedirs(save_dir + '/training_loss')
    os.makedirs(save_dir + '/model')
    os.makedirs(save_dir + '/tensorboard')

# create env
env = gym.make('PendulumDisturbance-v0')
env = env.unwrapped

# set seed
env.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

# observation/action dimensions
observation_dim = env.observation_space.shape[0]
action_dim      = env.action_space.shape[0]

# observation history dimensions
history_time_span       = 1  # consider 1s history observation/action
history_steps           = int(history_time_span/env.dt)
observation_history_dim = (observation_dim + action_dim)*history_steps + observation_dim

# model input/output dimensions
input_size  = observation_history_dim + action_dim
output_size = observation_dim

# sampling parameters
num_eps_train = 8000  # number of episodes to collect for training dataset
num_eps_val   = 1000  # number of episodes to collect for validation dataset
num_ep_steps  = 200   # number of steps in each episode

# neural network training parameters
num_layers      = 1    # number of hidden layers in dynamics model
num_layer_nodes = 500  # dimension of each hidden layer in dynamics model
batch_size      = 512
learning_rate   = 1e-3
num_epoches     = 20  # number of epoches for training the NN dynamics model

##################################################
###### Training/Validation Data Collection #######
##################################################

def collect_samples(num_eps):

    # render or not
    RENDER = False

    # data list of all episodes
    ep_observations_list            = []
    ep_observation_histories_list   = []
    ep_actions_list                 = []
    ep_next_observations_list       = []
    ep_observation_differences_list = []

    for i in range(num_eps):
        # number of episodes
        ep = i + 1

        # initial observation
        observation = env.reset()

        # initial observation history
        observation_history = observation

        # disturbance torque parameters
        A     = 2 + np.random.random()*1  # amplitude [2,3]
        T     = 2 + np.random.random()*2  # period [2,4]
        omega = 2*np.pi/T  # angular frequency
        phi   = np.random.random()*np.pi  # phase

        # data of each episode
        ep_observations            = []
        ep_observation_histories   = []
        ep_actions                 = []
        ep_next_observations       = []
        ep_observation_differences = []

        for j in range(num_ep_steps):
            # number of steps in an episode
            ep_step = j + 1

            # time in an episode
            ep_t = (ep_step - 1)*env.dt

            # render
            if RENDER and ep % 10 == 0:
                env.render()

            # disturbance torque (randomly change each episode)
            disturbance = A*np.sin(omega*ep_t + phi)

            # control torque (randomly choose)
            action = np.array([np.random.random()*env.max_torque*2 - env.max_torque])

            # take step
            next_observation, reward, done, info = env.step(action, disturbance)

            # change in observation over one time step
            observation_difference = next_observation - observation

            # observation history
            # when history observation space is not full
            if ep_step <= history_steps:
                # fill in observation history
                observation_history = np.concatenate((observation_history, action, next_observation))

            # when history observation space is full
            if ep_step > history_steps:
                # collect training data in each episode
                ep_observations.append(observation)
                ep_observation_histories.append(observation_history)
                ep_actions.append(action)
                ep_next_observations.append(next_observation)
                ep_observation_differences.append(observation_difference)

                # update observation history
                index = list(range(0, observation_dim + action_dim))
                observation_history = np.delete(observation_history, index)
                observation_history = np.concatenate((observation_history, action, next_observation))

            # update observation
            observation = next_observation

            if done or ep_step == num_ep_steps:
                if ep % 500 == 0:
                    print('Episode: %5i    Steps: %3i' % (ep, ep_step))
                break

        # data of each episode
        ep_observations            = np.array(ep_observations)
        ep_observation_histories   = np.array(ep_observation_histories)
        ep_actions                 = np.array(ep_actions)
        ep_next_observations       = np.array(ep_next_observations)
        ep_observation_differences = np.array(ep_observation_differences)

        # data list of all episodes
        # return list of length = num of episodes
        # each entry of that list contains one episode
        # each entry is [steps_per_episode x observation_history_dim] or [steps_per_episode x action_dim] or [steps_per_episode x observation_dim]
        ep_observations_list.append(ep_observations)
        ep_observation_histories_list.append(ep_observation_histories)
        ep_actions_list.append(ep_actions)
        ep_next_observations_list.append(ep_next_observations)
        ep_observation_differences_list.append(ep_observation_differences)

    # turn the list of episodes into just one large array of data
    sampled_observations            = np.concatenate(ep_observations_list, axis=0)
    sampled_observation_histories   = np.concatenate(ep_observation_histories_list, axis=0)
    sampled_actions                 = np.concatenate(ep_actions_list, axis=0)
    sampled_next_observations       = np.concatenate(ep_next_observations_list, axis=0)
    sampled_observation_differences = np.concatenate(ep_observation_differences_list, axis=0)

    return sampled_observations, sampled_observation_histories, sampled_actions, sampled_next_observations, sampled_observation_differences

print("\n##################################################")
print("Performing %5i episodes to collect training data" % num_eps_train)
print("##################################################\n")

# collect traing data
observations_train, observation_histories_train, actions_train, next_observations_train, observation_differences_train = collect_samples(num_eps_train)

print("\n##################################################")
print("Performing %5i episodes to collect validation data" % num_eps_val)
print("##################################################\n")

# collect validation data
observations_val,   observation_histories_val,   actions_val,   next_observations_val,   observation_differences_val   = collect_samples(num_eps_val)

print("\n##################################################")
print("Saving training/validation data")
print("##################################################\n")

# save training data
np.save(save_dir + '/training_data/observations.npy', observations_train)
np.save(save_dir + '/training_data/observation_histories.npy', observation_histories_train)
np.save(save_dir + '/training_data/actions.npy', actions_train)
np.save(save_dir + '/training_data/next_observations.npy', next_observations_train)
np.save(save_dir + '/training_data/observation_differences.npy', observation_differences_train)

# save validation data
np.save(save_dir + '/validation_data/observations.npy', observations_val)
np.save(save_dir + '/validation_data/observation_histories.npy', observation_histories_val)
np.save(save_dir + '/validation_data/actions.npy', actions_val)
np.save(save_dir + '/validation_data/next_observations.npy', next_observations_val)
np.save(save_dir + '/validation_data/observation_differences.npy', observation_differences_val)

print("Done collecting data")
print("Number of training data points: ", observations_train.shape[0])
print("Number of validation data points: ", observations_val.shape[0])

##################################################
############# Dynamic Model Training #############
##################################################

print("\n##################################################")
print("Preparing for dynamic model training")
print("##################################################\n")

# model input and label
# concatenate observation histories and actions as model input, to be used for training model
model_input_train = np.concatenate((observation_histories_train, actions_train), axis=1)
model_label_train = observation_differences_train

model_input_val   = np.concatenate((observation_histories_val, actions_val), axis=1)
model_label_val   = observation_differences_val

# combine training and validation dataset to calculate mean and std
# can also calculate mean and std of training dataset
model_input = np.concatenate((model_input_train, model_input_val), axis=0)
model_label = np.concatenate((model_label_train, model_label_val), axis=0)

# calculate mean value and standard deviation for the whole dataset
input_mean = np.mean(model_input, axis=0)
input_std  = np.std(model_input, axis=0)

label_mean = np.mean(model_label, axis=0)
label_std  = np.std(model_label, axis=0)

# tensorflow session
sess = tf.Session()

# initialize dynamic model
dynamic_model = DynamicModel(sess, save_dir, input_size, output_size, input_mean, input_std, label_mean, label_std,
                             num_layers, num_layer_nodes, batch_size, learning_rate, dtype=tf.float64)

# initialize all tensorflow variables
sess.run(tf.global_variables_initializer())

# make saver
saver = tf.train.Saver(max_to_keep=0)

print("\n##################################################")
print("Training the dynamic model")
print("##################################################\n")

# train dynamic model
training_loss, training_model_error = dynamic_model.train_model(model_input_train, model_label_train, observations_train, next_observations_train, num_epoches)

print("Training loss: %.9f" % training_loss)
print("Training model error rate: %3.3f %%\n" % (training_model_error*100))

# save trained model
save_path = saver.save(sess, save_dir + '/model/trained_dynamic_model.ckpt')

print("Model saved at ", save_path)

##################################################
############ Dynamic Model Validation ############
##################################################

### one-step validation is enough for our problem ###

print("\n##################################################")
print("Validating the dynamics model")
print("##################################################\n")

# run dynamic model to get loss
validation_loss, validation_model_error = dynamic_model.validate_model(model_input_val, model_label_val, observations_val, next_observations_val)

print("Validation loss: %.9f" % validation_loss)
print("validation model error rate: %3.3f %%\n" % (validation_model_error*100))

##################################################
##################### Timer ######################
##################################################

# end time
end = time.time()

# time used
elapsed = end - start

h = elapsed // 3600
m = (elapsed - h*3600) // 60
s = elapsed - h*3600 - m*60

print("Time used: %2i h - %2i m - %2i s" % (h, m, s))

