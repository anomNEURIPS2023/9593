"""A wrapper env that handles multiple tasks from different envs.
Useful while training multi-task reinforcement learning algorithms.
It provides observations augmented with one-hot representation of tasks.
"""

import random
import metaworld

class MultiEnvWrapper():
    def __init__(self, env_name, seed):


        self._num_tasks = 1
        self._active_task_index = None
        self._env_name = env_name
        import random
        mT10 = metaworld.MT10()
        self._env = mT10.train_classes[self._env_name]()
        self._task_list = [task for task in mT10.train_tasks if task.env_name == self._env_name]
        self._max_episode_steps = 150

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.render = self._env.render
    # @property
    # def observation_space(self):
    #     return self._env.observation_space
    #
    # @property
    # def action_space(self):
    #     return self._env.action_space

    def reset(self):
        self._env.set_task(random.choice(self._task_list))
        obs = self._env.reset()
        return obs

    def step(self, action):
        """Step the active task env.
        Args:
            action (object): object to be passed in Environment.reset(action)
        Returns:
            EnvStep: The environment step resulting from the action.
        """
        state, reward, done, info = self._env.step(action)
        return state, reward, done, info

    def close(self):
        self._env.close()

    def seed(self, seed):
        self._env.seed(seed)
