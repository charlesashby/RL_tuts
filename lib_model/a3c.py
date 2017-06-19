import tensorflow as tf
from lib.ops import conv2d, fc_layer
from lib.environment import Environment
import numpy as np

PATH = '/home/ashbylepoc/PycharmProjects/RL/'
MAX_TIME_STEP = 10 * 10 ** 7
LOG_FILE = PATH + 'checkpoints/log.txt'
CHECKPOINT_DIR = PATH + 'checkpoints'

class A3C(object):
    """ A3C Model Implementation """

    def __init__(self, action_size,
                 thread_index, device,
                 entropy_beta=0.001
                 ):
        """
        :param action_size:
            number of actions the agent can take

        :param thread_index:
            Index of the agent (-1 for global)

        :param device:
            Device used e.g. '/cpu:0'
        """
        self._device = device
        self._action_size = action_size
        self._thread_index = thread_index
        self.entropy_beta = entropy_beta

    def _create_a3c_network(self):
        """ Creates the A3C network """
        # Input image is of shape [84 x 84 x 3]
        self.input = tf.placeholder("float", [None, 84, 84, 3])

        # The action is a one-hot encoded vector of shape [self._action_size]
        # and the reward is a floating point. We return both values as a
        # concatenated vector of shape [self._action_size + 1]
        self.last_action_reward = tf.placeholder("float",
                                                 [None, self._action_size + 1])

        # We use the same network as Mnih & Al.'s A3C implementation
        # [batch_size x 20 x 20 x 16]
        cnn = conv2d(self.input, 16, 8, 8, stride=4, name='conv0')

        # [batch_size x 9 x 9 x 32]
        cnn = conv2d(cnn, 32, 4, 4, stride=2, name='conv1')

        # we reshape the output of the conv layer to [batch_size x 32 * 9 * 9]
        # lstm_input is of shape [batch_size x 256], note that in our case
        # the batch_size is the number of frames. In our implementation
        # we will compute the outputs of the LSTM frame-by-frame and backpropagate
        # every 20 frames, thus, sequence_length (forward) = 1 and sequence_length
        # (backward) = 20
        lstm_input = fc_layer(tf.reshape(cnn, [-1, 2592]), 256, name='fc0')
        # sequence_length = tf.shape(lstm_input)[:1]

        with tf.variable_scope('lstm') as scope:
            # In the paper, they concatenate the downsampled environment
            # with the last action and reward before feeding it to the LSTM
            lstm_input = tf.concat([lstm_input, self.last_action_reward], 1)

            # the dynamic_rnn method takes an input of shape
            # [batch_size x sequence_length x input_dim] in our case
            # batch_size = 1, sequence_length = unroll_step (default:20)
            # and input_dim = 256 + action_size + 1 (lstm_input + last action
            # encoded as one-hot vector + the reward (float)
            lstm_input = tf.reshape(lstm_input, [1, -1, 256 + self._action_size + 1])

            # The LSTM cell is created in the _create_network method,
            # here we only initialize it
            initial_state = self.lstm_cell.zero_state(batch_size=1, dtype=tf.float32)

            # Fetch the output and the last state of the LSTM, Given the cell
            # state of an LSTM and the input at time t we can compute the
            # output and cell state at time t + 1 (t = 0, 1, ...), therefore, we use
            # the state to forward propagate manually. This will become
            # clear once we get into the actual training
            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                                   lstm_input,
                                                                   initial_state=initial_state,
                                                                   scope=scope,
                                                                   dtype=tf.float32
                                                                  )

            # self.lstm_outputs is of shape [batch_size=1 x seq_length x n_units] we
            # simply reshape it to [seq_length x n_units]
            self.lstm_outputs = tf.reshape(self.lstm_outputs, shape=[-1, 256])

        # Once we have the output of the LSTM we need to compute the policy (pi)
        # and the value function (v) for this frame, both of them are
        # approximated using a neural network

        # pi is of shape [batch_size=1, self._action_size] it is
        # the probability distribution from which the action
        # are sampled
        with tf.variable_scope('policy') as scope:
            self.pi = fc_layer(self.lstm_outputs, self._action_size,
                          name='fc_pi', activation=tf.nn.softmax)

        # v is of shape [batch_size=1, 1] (floating point)
        with tf.variable_scope('value') as scope:
            self.v = fc_layer(self.lstm_outputs, 1, name='fc_v',
                         activation=None)

    def _create_network(self):
        # Scope name for the network, A3C() can create a
        # network for an agent or for the main network; the
        # one that is updated asynchronously.
        scope_name = "net_{0}".format(self._thread_index)

        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # We share the LSTM cell across the whole network, therefore,
            # we instantialize it before the A3C network this will be
            # useful when we start adding the unsupervised auxiliary tasks
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,
                                                          state_is_tuple=True)

            # Create the A3C network...
            self._create_a3c_network()

            # Get the variables of the model
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope_name)

            # Reset the state (mostly for creating lstm_state_out)
            self.reset_state()

    def get_vars(self):
        """ Get the variables of the network """
        return self.variables

    def reset_state(self):
        """
        Method to reset the state of the LSTM
        once an episode has been terminated
        """
        # Note: Remember that the state != weights!
        # This method is equivalent to clearing the memory of the
        # lstm layer
        # self.lstm_state is a tuple of shape ([1, 256], [1,256]). We
        # need to initialize it with all zeros
        self.lstm_state_out = (np.zeros([1, 256], dtype='float32'),
                               np.zeros([1, 256], dtype='float32'))


    def prepare_a3c_loss(self):
        """
        A3C losses: (One for updating the parameters of the policy
            network and the other for updating the parameters of
            the value function network - the base network is
            updated by taking the sum of both)

        policy loss: SUM_{t = t_start}^{T}[log(pi(a_t) * a_t
                                        * (R(s_t) - v(s_t))) + H(pi(a_t)]

        value function loss: SUM_{t = t_start}^{T}[(R(s_t) - v(s_t))^2]

        total loss: policy loss + value function loss

        :pi(a_t):
            probability of the action taken at time step t (a_t)

        :R(s_t):
            Reward at time t. Equal to 0 if s_t is a terminal state
            else v(s_t) if s_t is not terminal

        :v(s_t):
            Value function for state s_t

        :a_t:
            Action that was taken at time step t
            remember that this is a one-hot vector

        :H(pi(a_t)):
            Entropy of the policy

        """

        # Placeholder for a_t; t = t_start, ..., T
        self.a = tf.placeholder(dtype=tf.float32,
                                shape=[None, self._action_size])

        # Placeholder for R(s_t); t = t_start, ..., T
        self.R = tf.placeholder(dtype=tf.float32,
                                shape=[None])

        # R(s_t) - v(s_t) is called the advantage function
        self.adv = tf.placeholder(dtype=tf.float32,
                                  shape=[None])

        # We clip log(pi(a_t)) to make sure we dont get NaN when
        # pi(a_t) is 0
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

        # log_pi has the same shape as self.a, that is
        # [batch_size x action_size]
        self.log_pi = tf.reduce_sum(tf.multiply(log_pi, self.a))

        # Note that in the algorithm in the original A3C paper,
        # does not use the entropy of pi, but it has been shown
        # to improve training stability
        self.entropy = - tf.reduce_sum(self.pi * self.log_pi, reduction_indices=1)

        # loss for the policy (actor)
        self.policy_loss = - tf.reduce_sum(self.log_pi) * self.adv \
                                + self.entropy * self.entropy_beta

        # Loss for the value function (critic); it is multiplied
        # by 0.5 because the learning rate should be half as the
        # one for the policy (actor)
        self.value_loss = 0.5 * tf.nn.l2_loss(self.R - self.v)

        # Loss for the base network (LSTM & frame-embedding)
        self.a3c_loss = self.policy_loss + self.value_loss

    def run_pi_value(self, sess, s_t, last_action_reward):
        """
        Method to forward propagate through the network
        frame-by-frame.

        :s_t:
            Last state of the ENVIRONMENT

        :last_action_reward:
            Last action taken concatenated
            with the last reward
        """
        pi_out, v_out, self.lstm_state_out = \
            sess.run([self.pi, self.v, self.lstm_state],
                      feed_dict={self.input: [s_t],
                                 self.last_action_reward: [last_action_reward],
                                 self.lstm_state: [self.lstm_state_out]})

        return pi_out.flatten(), v_out.flatten()

    def run_last_value(self, sess, s_t, last_action_reward):
        """
        Note that while this looks like run_pi_value(), here, we do
        not update the lstm_state since this method is used at the
        end of local_t_max time steps and the state is updated at the
        beginning of the loop. This will get clear when we build the
        actual Trainer() class
        """
        v_out, _ = sess.run([self.v, self.lstm_state],
                             feed_dict={self.input: [s_t],
                                        self.last_action_reward: [last_action_reward],
                                        self.lstm_state: self.lstm_state_out})

        return v_out.flatten()

    def run_losses(self, sess, feed_dict):
        """ Compute the losses for logging/printing """
        total_loss, value_loss, policy_loss = \
            sess.run([self.a3c_loss, self.value_loss, self.policy_loss],
                     feed_dict=feed_dict)
        return total_loss, value_loss, policy_loss

    def sync_from(self, src_network, name=None):
        """
        Update the variables of the network; used when
        starting a new episode with the global network
        """
        # Source variables
        src_vars = src_network.variables

        # Destination variables
        dst_vars = self.variables

        # Syncing ops
        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "A3CModel", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_ops.append(tf.assign(dst_var, src_var))

            return tf.group(*sync_ops, name=name)


class Trainer(object):
    """ Class for Training a Local Network / ONE agent """

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate,
                 grad_applier,
                 local_t_max=20,
                 max_global_time_step=10 * 10**7,
                 gamma=0.99,
                 save_interval_step=100 * 1000,
                 env='Breakout-v0',
                 device='/cpu:0'):

        self.thread_index = thread_index
        self.learning_rate = learning_rate
        self.env = env

        # Discount factor for the reward
        self.gamma = gamma

        # Number of "epochs"
        self.max_global_time_step = max_global_time_step

        # Number of steps for the LSTM
        self.local_t_max = local_t_max

        # Number of actions the agent can take
        self.action_size = Environment.get_action_size(env)

        self.local_network = A3C(self.action_size,
                                 self.thread_index,
                                 device)

        self.global_network = global_network

        # Build computational graph
        self.local_network._create_network()

        # Build computational graph for the losses
        # and gradients
        self.local_network.prepare_a3c_loss()
        self.apply_gradients = grad_applier.minimize_local(self.local_network.a3c_loss,
                                                           global_network.get_vars(),
                                                           self.local_network.get_vars())

        # Sync the weights of the local network with those
        # of the main network
        self.sync = self.local_network.sync_from(global_network)

        # Initialize time step, learning rate, etc
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0

    def build_environment(self):
        """ Create the environment """
        self.environment = Environment(self.env)

    def stop(self):
        """ Terminate the environment """
        self.environment.stop()

    def choose_action(self, pi_values):
        """
        Sample from the learned policy
        distribution

        :param pi_values:
            Probability distribution for
            every actions
        """
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def concat_action_reward(self, action, action_size, reward):
        """
        Return one hot vectored action and reward.
        """
        action_reward = np.zeros([action_size + 1], dtype='float32')
        action_reward[action] = 1.0
        action_reward[-1] = float(reward)
        return action_reward

    def _decay_learning_rate(self, global_time_step):
        """ Decay the learning rate linearly """
        time_left = self.max_global_time_step - global_time_step
        learning_rate = self.initial_learning_rate * time_left \
                        / self.max_global_time_step

        # Clip learning rate at 0.0
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def _process_a3c(self, sess, global_t):
        """
        Process max_local_t steps/frames in the
        A3C network

        :param sess:
            TensorFlow session object

        :param global_t:
            Global time step (number of steps
            processed by the global/shared network)
        """

        states = []
        last_action_rewards = []
        actions = []
        rewards = []
        values = []
        
        # Initialize local_t
        self.local_t = 0
        
        # Synchronize with global network
        sess.run(self.sync)
        
        # Whether we hit a terminal state or not
        terminal_end = False
        start_lstm_state = self.local_network.lstm_state_out

        # Loops local_t_max time steps
        for _ in range(self.local_t_max):
            last_action = self.environment.last_action
            last_reward = self.environment.last_reward
            last_action_reward = self.concat_action_reward(last_action,
                                                           self.action_size,
                                                           last_reward)

            # Compute policy and value function
            pi_, value_ = self.local_network.run_pi_value(sess,
                                                  self.environment.last_state,
                                                  last_action_reward)

            # Pick an action given the new computed policy
            action = self.choose_action(pi_)

            # Append results to placeholders...
            states.append(self.environment.last_state)
            last_action_rewards.append(last_action_reward)
            actions.append(action)
            values.append(value_)

            prev_state = self.environment.last_state

            # Process next action
            new_state, reward, terminal = self.environment.process(action)

            rewards.append(reward)
            self.episode_reward += reward

            self.local_t += 1

            if terminal:
                # Environment hit a terminal state
                terminal_end = True

                # If we hit a terminal state, then the
                # reward is set to 0, else, it is set
                # to the value function
                self.episode_reward = 0
                self.environment.reset()
                self.local_network.reset_state()
                break

        # ---------
        # BACK-PROP
        # ---------

        # We discount the rewards from t - 1 to t_start. At
        # time step t the reward is either 0 (if terminal state)
        # or V (non terminal state)
        R = 0.0
        if not terminal_end:
            R = self.local_network.run_last_value(sess,
                                                  new_state,
                                                  last_action_reward)

        # Reverse placeholders
        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        # To compute the gradients we compute a minibatch of
        # length local_t_max
        batch_s = []
        batch_a = []
        batch_adv = []
        batch_R = []



        # Discounting...
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + self.gamma * R
            adv = R - Vi
            a = np.array([0] * self.action_size)
            a[ai] = 1.0


            batch_s.append(si)
            batch_a.append(a)

            # Convert np.array -> float because
            # the advantage and reward placeholders
            #  expects shape [None, ] not [None, 1]
            batch_adv.append(float(adv))
            batch_R.append(float(R))

        batch_s.reverse()
        batch_a.reverse()
        batch_adv.reverse()
        batch_R.reverse()

        # Decay learning rate
        start_local_t = self.local_t
        cur_learning_rate = self._decay_learning_rate(global_t)

        # Create feed_dict for gradient_applier
        feed_dict = {
            self.local_network.input: batch_s,
            self.local_network.last_action_reward: last_action_rewards,
            self.local_network.a: batch_a,
            self.local_network.adv: batch_adv,
            self.local_network.R: batch_R,
            self.local_network.lstm_state: start_lstm_state,
            self.learning_rate: cur_learning_rate
        }


        # compute gradients and update weights
        sess.run(self.apply_gradients, feed_dict=feed_dict)
        """
        # ----------------
        # PRINT STATISTICS
        # ----------------

        # Compute losses
        total_loss, policy_loss, value_loss = self.local_network.run_losses(sess,
                                                                            feed_dict)

        total_loss = np.mean(total_loss)
        policy_loss = np.mean(policy_loss)
        value_loss = np.mean(value_loss)

        if global_t % 1000 == 0:
            print('Reward: %3d - Total Loss: %.4f - Policy Loss: %.4f '
                  '- Value Loss: %.4f' %
                  (float(R), total_loss, policy_loss, value_loss))

            # Save to log file
            with open(LOG_FILE, 'a') as f:
                f.write('Reward: %3d - Total Loss: %.4f - Policy Loss: %.4f '
                  '- Value Loss: %.4f \n' %
                  (float(R), total_loss, policy_loss, value_loss))
        """
        
        # Return the number of steps taken
        # to update global_time_steps
        return self.local_t

if __name__ == '__main__':
    network = A3C(action_size=5,
                  thread_index=1,
                  device='/cpu:0')
    network._create_network()
