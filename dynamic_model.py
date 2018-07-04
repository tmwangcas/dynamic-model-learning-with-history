import numpy as np
import tensorflow as tf
import math
import time

class DynamicModel:

    def __init__(self, sess, save_dir, input_size, output_size, input_mean, input_std, label_mean, label_std,
                 num_layers, num_layer_nodes, batch_size, learning_rate, dtype):

        # initilize variables
        self.sess       = sess
        self.save_dir   = save_dir
        self.input_mean = input_mean
        self.input_std  = input_std
        self.label_mean = label_mean
        self.label_std  = label_std
        self.batch_size = batch_size

        # placeholders
        self.nn_input_ph = tf.placeholder(dtype=dtype, shape=[None, input_size],  name='neutal_network_input')
        self.nn_label_ph = tf.placeholder(dtype=dtype, shape=[None, output_size], name='neutal_network_label')

        self.curr_obs_true_ph = tf.placeholder(dtype=dtype, shape=[None, output_size], name='current_observation_true')
        self.next_obs_true_ph = tf.placeholder(dtype=dtype, shape=[None, output_size], name='next_observation_true')

        # build neural network
        self.nn_output = self.build_neural_network(self.nn_input_ph, output_size, num_layers, num_layer_nodes, dtype=dtype, reuse=False)

        # model output (observation difference)
        self.model_output = self.nn_output*self.label_std + self.label_mean

        # prediction value of next observation
        self.next_obs_pred = self.model_output + self.curr_obs_true_ph

        # loss function (mean square error of neural network output)
        self.nn_mse = tf.reduce_mean(tf.square(self.nn_label_ph - self.nn_output))

        # mean error rate of model prediction
        self.model_error = tf.reduce_mean(tf.abs((self.next_obs_true_ph - self.next_obs_pred)/self.next_obs_true_ph))

        # compute gradients and update parameters
        self.opt        = tf.train.AdamOptimizer(learning_rate)
        self.theta      = tf.trainable_variables()
        self.gv         = [(g,v) for g,v in
                           self.opt.compute_gradients(self.nn_mse, self.theta)
                           if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

        # tensorboard setting
        tf.summary.scalar('neural_network_mean_square_error', self.nn_mse)
        tf.summary.scalar('model_prediction_error_rate', self.model_error)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.save_dir + '/tensorboard/', self.sess.graph)

        # run in terminal:
        # tensorboard --logdir your_directory/saved_data/tensorboard/


    def train_model(self, model_input, model_label, curr_obs_true, next_obs_true, num_epoches):

        # training start time
        train_start = time.time()

        # preprocess model input and label data (scaled)
        nn_input = np.nan_to_num((model_input - self.input_mean)/self.input_std)
        nn_label = np.nan_to_num((model_label - self.label_mean)/self.label_std)

        # initilize variables
        num_data_points        = nn_input.shape[0]
        range_of_indices       = np.arange(num_data_points)
        num_epoch_batches      = int(math.floor(num_data_points/self.batch_size))
        num_batches            = 0
        epoch_loss             = 0
        batch_loss_list        = []
        epoch_loss_list        = []
        epoch_model_error      = 0
        batch_model_error_list = []
        epoch_model_error_list = []

        # training loop
        for i in range(num_epoches):
            # reset total batch loss and model error in each epoch
            total_batch_loss        = 0
            total_batch_model_error = 0

            # randomly order indices (equivalent to shuffling input and label)
            random_indices = np.random.choice(range_of_indices, size=(num_data_points,), replace=False)

            # get through the full dataset
            for batch in range(num_epoch_batches):
                # walk through the randomly reordered data
                nn_input_batch = nn_input[random_indices[batch*self.batch_size : (batch + 1)*self.batch_size], :]
                nn_label_batch = nn_label[random_indices[batch*self.batch_size : (batch + 1)*self.batch_size], :]

                curr_obs_true_batch = curr_obs_true[random_indices[batch*self.batch_size : (batch + 1)*self.batch_size], :]
                next_obs_true_batch = next_obs_true[random_indices[batch*self.batch_size : (batch + 1)*self.batch_size], :]

                # one iteration of feedforward training
                _, batch_loss, batch_model_error = self.sess.run([self.train_step, self.nn_mse, self.model_error],
                                                                 feed_dict={self.nn_input_ph:      nn_input_batch,
                                                                            self.nn_label_ph:      nn_label_batch,
                                                                            self.curr_obs_true_ph: curr_obs_true_batch,
                                                                            self.next_obs_true_ph: next_obs_true_batch})

                # tensorboard
                num_batches += 1
                result = self.sess.run(self.merged, feed_dict={self.nn_input_ph:      nn_input_batch,
                                                               self.nn_label_ph:      nn_label_batch,
                                                               self.curr_obs_true_ph: curr_obs_true_batch,
                                                               self.next_obs_true_ph: next_obs_true_batch})
                self.writer.add_summary(result, num_batches)

                # sum up all batch loss and model error
                total_batch_loss += batch_loss
                batch_loss_list.append(batch_loss)

                total_batch_model_error += batch_model_error
                batch_model_error_list.append(batch_model_error)

            # loss and model error of each epoch
            epoch_loss = total_batch_loss/num_epoch_batches
            epoch_loss_list.append(epoch_loss)

            epoch_model_error = total_batch_model_error/num_epoch_batches
            epoch_model_error_list.append(epoch_model_error)

            print("===== Epoch {} =====".format(i+1))
            print("epoch loss: %.9f" % epoch_loss)
            print("epoch model error rate: %3.3f %%\n" % (epoch_model_error*100))

        # save loss and model error after training
        np.save(self.save_dir + '/training_loss/batch_loss.npy', batch_loss_list)
        np.save(self.save_dir + '/training_loss/epoch_loss.npy', epoch_loss_list)

        np.save(self.save_dir + '/training_loss/batch_model_error.npy', batch_model_error_list)
        np.save(self.save_dir + '/training_loss/epoch_model_error.npy', epoch_model_error_list)

        print("Training set size: ", num_data_points)

        # training end time
        train_end = time.time()

        print("Training duration: {:0.2f} s".format(train_end - train_start), "\n")

        # loss and model error of the final epoch
        training_loss        = epoch_loss
        training_model_error = epoch_model_error

        return training_loss, training_model_error


    def validate_model(self, model_input, model_label, curr_obs_true, next_obs_true):

        # preprocess model input and label data (scaled)
        nn_input = np.nan_to_num((model_input - self.input_mean)/self.input_std)
        nn_label = np.nan_to_num((model_label - self.label_mean)/self.label_std)

        # run the model to get the loss and model error
        validation_loss, validation_model_error = self.sess.run([self.nn_mse, self.model_error], feed_dict={self.nn_input_ph:      nn_input,
                                                                                                            self.nn_label_ph:      nn_label,
                                                                                                            self.curr_obs_true_ph: curr_obs_true,
                                                                                                            self.next_obs_true_ph: next_obs_true})

        return validation_loss, validation_model_error


    # build neural network
    @staticmethod
    def build_neural_network(nn_input, output_size, num_layers, num_layer_nodes, dtype, reuse):

        # variables
        hidden_size = num_layer_nodes
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=dtype)
        fc = tf.contrib.layers.fully_connected

        # make hidden layers
        for i in range(num_layers):
            if i == 0:
                fc_i = fc(nn_input, num_outputs=hidden_size, activation_fn=None, weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
            else:
                fc_i = fc(h_i, num_outputs=hidden_size, activation_fn=None, weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
            h_i = tf.nn.relu(fc_i)

        # make output layer
        nn_output = fc(h_i, num_outputs=output_size, activation_fn=None, weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)

        return nn_output
