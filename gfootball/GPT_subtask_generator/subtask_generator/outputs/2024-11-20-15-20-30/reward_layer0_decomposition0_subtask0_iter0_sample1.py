import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a specific reward function for focusing on technical skills such as passing, shooting, and dribbling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_reward = 0.2
        self.shot_reward = 0.3
        self.dribble_reward = 0.1
        self.control_bonus = 0.05

    def reset(self):
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        reward_components = {"base_score_reward": reward.copy(),
                             "pass_reward": [0.0] * len(reward),
                             "shot_reward": [0.0] * len(reward),
                             "dribble_reward": [0.0] * len(reward),
                             "control_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, reward_components
        
        for i in range(len(reward)):
            o = observation[i]
            # Encourage passing by giving a reward every time a pass is successfully completed
            if o['sticky_actions'][6] or o['sticky_actions'][7] or o['sticky_actions'][8]:
                reward_components["pass_reward"][i] = self.pass_reward

            # Encourage shooting at the goal
            if o['sticky_actions'][9]:  # Assuming action 9 corresponds to `Shot`
                reward_components["shot_reward"][i] = self.shot_reward
            
            # Encourage dribbling capability
            if o['sticky_actions'][10]:  # Assuming action 10 corresponds to `Dribble`
                reward_components["dribble_reward"][i] = self.dribble_reward

            # Control bonus for keeping the ball
            if o['ball_owned_player'] == i and o['ball_owned_team'] == 0:  # Assume team 0 is the controlled team
                reward_components["control_bonus"][i] = self.control_bonus

            # Sum the rewards
            reward[i] += sum(reward_components[item][i] for item in reward_components)

        return reward, reward_components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        
        # Modify the reward using the reward method
        modified_reward, components = self.reward(reward)
        
        # Add final reward to the info
        info["final_reward"] = sum(modified_reward)
        
        # Include components in the info for debugging
        for key, value in components.items():
            info["component_" + key] = sum(value)
        
        return observation, modified_reward, done, info
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
