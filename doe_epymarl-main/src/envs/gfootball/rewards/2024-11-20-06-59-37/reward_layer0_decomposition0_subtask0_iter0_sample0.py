import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effectively developing offensive skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize any necessary variables here
        self.pass_rewards = 0.2
        self.shot_rewards = 0.5
        self.dribble_rewards = 0.3

    def reset(self):
        # Reset any variables if necessary
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Modify the rewards based on offensive actions taken
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Encourage passing when strategy benefits the team
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                components['pass_reward'][rew_index] = self.pass_rewards
                reward[rew_index] += components['pass_reward'][rew_index]
            
            # Reward for successful shots on goal (simply approximate here by checking shot action)
            if o['sticky_actions'][8] == 1: # Assuming index 8 is shot action
                components['shot_reward'][rew_index] = self.shot_rewards
                reward[rew_index] += components['shot_reward'][rew_index]
            
            # Encourage dribbling to maneuver around opponents
            if o['sticky_actions'][6] == 0: # Assuming index 6 is dribble action
                components['dribble_reward'][rew_index] = self.dribble_rewards
                reward[rew_index] += components['dribble_reward'][rew_index]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
