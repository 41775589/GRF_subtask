import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for mastering skills related to controlling the game tempo and maintaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self.passing_reward = 0.1
        self.movement_reward = 0.05
        self.defensive_action_reward = 0.2
        self.pass_types = {'Short Pass', 'High Pass', 'Long Pass'}

    def reset(self):
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "movement_reward": [0.0] * len(reward),
                      "defensive_action_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for executing pass types effectively
            if o['sticky_actions'][7] in self.pass_types:
                components["pass_reward"][rew_index] = self.passing_reward

            # Reward for effective movement to strategic positions
            if any(o['sticky_actions'][:8]):  # Considering first 8 are movement actions
                components["movement_reward"][rew_index] = self.movement_reward

            # Reward for defensive actions (e.g., Sliding to regain possession)
            if o['sticky_actions'][5]:  # Assuming index 5 is for Sliding
                components["defensive_action_reward"][rew_index] = self.defensive_action_reward
            
            # Sum up all component rewards to the original reward
            reward[rew_index] += (components["pass_reward"][rew_index] +
                                  components["movement_reward"][rew_index] +
                                  components["defensive_action_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
