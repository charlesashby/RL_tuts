# Modified from https://github.com/miyosuda/unreal/
# utils for playing around in the Gym environments
# with multiprocessing

from multiprocessing import Process, Pipe
import numpy as np
import cv2
import gym

COMMAND_RESET = 0
COMMAND_ACTION = 1
COMMAND_TERMINATE = 2


def preprocess_frame(observation):
    # observation shape = (210, 160, 3)
    observation = observation.astype(np.float32)
    resized_observation = cv2.resize(observation, (84, 84))
    resized_observation /= 255.0
    return resized_observation


def worker(conn, env_name):
    """

    :param conn:
        Pipe() object for multiprocess

    :param env_name:
        Name of the environment, in
        our case we use 'Breakout-v0'
    """
    # Create the Gym environment
    env = gym.make(env_name)

    # Reset the environment
    env.reset()

    # conn is the child end of a Pipe() object it
    # sends data received by the parent's. For more
    # info on Pipes and multiprocessing in general
    # see https://docs.python.org/2/library/multiprocessing.html
    conn.send(0)

    while True:
        # Receive the command
        command, arg = conn.recv()

        # Commands are either 0, 1 or 2,
        # respectively: RESET, ACTION or
        # TERMINATE
        if command == COMMAND_RESET:
            # Reset the environment, reshape
            # it and send it
            obs = env.reset()
            state = preprocess_frame(obs)
            conn.send(state)

        elif command == COMMAND_ACTION:
            # In Gym, every actions are
            # repeated 4 times. We compute
            # the reward and whether the
            # ending state was terminal or not
            reward = 0
            for i in range(4):
                obs, r, terminal, _ = env.step(arg)
                reward += r
                if terminal:
                    break
            state = preprocess_frame(obs)
            conn.send([state, reward, terminal])

        elif command == COMMAND_TERMINATE:
            # Terminate the environment
            break

        else:
            print("bad command: {}".format(command))

    # When we get out of the main loop, we get ready
    # to reset the environment
    env.close()
    conn.send(0)
    conn.close()


class Environment(object):
    """
    Class that controls ONE environment (ONE agent), you
    can get the process ID with:
    import os; os.getpid()
    """
    @staticmethod
    def get_action_size(env_name):
        # Get the number of actions that
        # the agent can do
        env = gym.make(env_name)
        action_size = env.action_space.n
        env.close()
        return action_size

    def __init__(self, env_name):
        # For more info on Pipes and multiprocessing in
        # general see https://docs.python.org/2/library/multiprocessing.html
        self.conn, child_conn = Pipe()

        # Our agents are controlled by the worker() method. They
        # receive commands from the parent connection that are
        # received by the child connection
        self.proc = Process(target=worker, args=(child_conn, env_name))
        self.proc.start()
        self.conn.recv()
        self.reset()

    def reset(self):
        # Reset the environment state
        self.conn.send([COMMAND_RESET, 0])
        self.last_state = self.conn.recv()
        self.last_action = 0
        self.last_reward = 0

    def stop(self):
        # Terminate the environment
        self.conn.send([COMMAND_TERMINATE, 0])
        ret = self.conn.recv()
        self.conn.close()
        self.proc.join()
        print("gym environment stopped")

    def process(self, action):
        # Do an action in the environment
        self.conn.send([COMMAND_ACTION, action])
        state, reward, terminal = self.conn.recv()

        # Those information will be fed to our
        # network and used for forward prop
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal

