import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a structured skill-based reward for ball control."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_accuracy_reward = 0.1
        self.dribble_proximity_reward = 0.2
        self.shooting_reward = 0.3
        
    def reset(self):
        """Resets the environment and the reward conditions."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Extracts the state for potential saving and replaying."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state if loading from previous save state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on subtask-specific skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_accuracy_reward": [0.0] * len(reward),
                      "dribble_proximity_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 0:
                # Reward for maintaining ball possession through accurate passes
                components["pass_accuracy_reward"][rew_index] = self.pass_accuracy_reward
                reward[rew_index] += components["pass_accuracy_reward"][rew_index]
                
                # Reward for successful dribbling improving proximity to goal
                if np.linalg.norm(o['ball'] - np.array([1, 0])) < 0.5:  # Assuming the goal at (1, 0)
                    components["dribble_proximity_reward"][rew_index] = self.dribble_proximity_reward
                    reward[rew_index] += components["dribble_proximity_reward"][rew_index]
                
                # Reward for shots directed towards the goal
                if 'action' in o and o['action'] == 'Shot':
                    components["shooting_reward"][rew_index] = self.shooting_reward
                    reward[rew_index] += components["shooting_reward"][rew_index]

        return reward, components
    
    def step(self, action):
        """Applies an action to the environment, adjusts the reward, and returns observation, reward, done, info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
