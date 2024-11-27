import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {i: 0 for i in range(3)}
        self._num_checkpoints = 3
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {i: 0 for i in range(3)}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            if reward[i] == 1:
                components["checkpoint_reward"][i] = self._checkpoint_reward * (
                        self._num_checkpoints - self._collected_checkpoints[i])
                reward[i] = 1 * components["base_score_reward"][i] + components["checkpoint_reward"][i]
                self._collected_checkpoints[i] = self._num_checkpoints
                continue

            # Implement checkpoint rewards based on the position of the ball or player for each agent
            # Here you can define the conditions to collect a checkpoint reward based on offensive actions

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
