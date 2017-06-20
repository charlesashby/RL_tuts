import tensorflow as tf
import numpy as np
import threading
import math

from lib.environment import Environment
from lib_model.a3c import A3C, Trainer
from lib.optimizer import RMSPropApplier

PATH = '/home/ashbylepoc/PycharmProjects/RL/'

# Most models from Deepmind trained for ~20 hours
# at 1000 steps/sec
MAX_TIME_STEP = 10 * 10 ** 7
LOG_FILE = PATH + 'checkpoints/log'
CHECKPOINT_DIR = PATH + 'checkpoints'


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

class App(object):
    """ Application to run and train everything """

    def __init__(self,
                 device='/cpu:0',
                 env='Breakout-v0',
                 initial_alpha_low=1e-4,
                 initial_alpha_high=5e-3,
                 parallel_size=8,
                 checkpoint_interval=100 * 1000,
                 logging_interval=1000):

        self.env = env
        self.device = device
        self.initial_alpha_low = initial_alpha_low
        self.initial_alpha_high = initial_alpha_high
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval

        # Number of agents
        self.parallel_size = parallel_size

    def train(self, parallel_index, preparing):
        """ This is our worker method for multiprocessing """

        # Agent to train; self.trainers is
        # defined in run() and is a placeholder
        # for all the agents
        trainer = self.trainers[parallel_index]

        # Create the environment
        if preparing:
            trainer.build_environment()

        # Main loop
        while True:
            if self.stop:
                break

            if self.terminate:
                trainer.stop()
                break

            if self.global_t > MAX_TIME_STEP:
                trainer.stop()
                break

            # Save the network
            if parallel_index == 0 and self.global_t > self.next_checkpoint:
                self.saver.save(self.sess, SAVE_PATH)

                # Set next checkpoint
                self.next_checkpoint += self.checkpoint_interval

            # Process local_t timesteps and update
            # global_t
            diff_global_t = trainer._process_a3c(self.sess, self.global_t,
                                                 self.summary_writer,
                                                 self.summary_op,
                                                 self.score_input)

            # Set global time step
            self.global_t += diff_global_t


    def run(self):
        """ Run the model """

        with tf.device(self.device):
            # The learning rate is sampled from a
            # log-uniform distribution between
            # 0.0001 and 0.005. Then, it is
            # decayed linearly to 0 progressively
            # during training
            initial_learning_rate = log_uniform(self.initial_alpha_low,
                                                self.initial_alpha_high,
                                                0.5)

            # Whether to terminate, pause or keep training
            self.stop = False
            self.terminate = False

            # Initialize global time step
            self.global_t = 0

            # Number of actions the agent can take
            action_size = Environment.get_action_size(self.env)

            # Initialize the shared/global network
            self.global_network = A3C(action_size,
                                      thread_index=-1,
                                      device=self.device)

            # Build computational graph
            self.global_network._create_network()

            # Placeholder for the Trainers
            self.trainers = []

            learning_rate_input = tf.placeholder("float")

            # Initialize the RMSPROP object for the updates
            grad_applier = RMSPropApplier(learning_rate_input)

            # Build the agents
            for i in range(self.parallel_size):
                trainer = Trainer(thread_index=i,
                                  global_network=self.global_network,
                                  initial_learning_rate=initial_learning_rate,
                                  grad_applier=grad_applier,
                                  learning_rate=learning_rate_input)
                if i == 0:
                    trainer.show_env = True

                self.trainers.append(trainer)

            # Initialize Session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            # Params for logging scores in Tensorboard
            self.score_input = tf.placeholder(tf.int32)
            tf.summary.scalar("score", self.score_input)
            self.summary_op = tf.summary.merge_all()

            # sess.graph contains the graph definition;
            # that enables the Graph Visualizer. To start
            # Tensorboard run the following command:
            # $ tensorboard --logdir=path/to/LOG_FILE
            self.summary_writer = tf.summary.FileWriter(LOG_FILE,
                                                        graph=self.sess.graph)


            # Parameters for saving the global network params
            self.saver = tf.train.Saver(var_list=self.global_network.get_vars(),
                                        max_to_keep=1)

            # Set next checkpoint
            self.next_checkpoint = self.checkpoint_interval

            # Set next log point
            self.next_log = self.logging_interval

            # -----------
            # RUN THREADS
            # -----------

            self.train_threads = []
            for i in range(self.parallel_size):
                self.train_threads.append(threading.Thread(target=self.train,
                                                           args=(i, True)))
            for t in self.train_threads:
                t.start()

if __name__ == '__main__':
    agent = App()
    agent.run()
