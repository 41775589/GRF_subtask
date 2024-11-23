import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specific to mastering individual technical skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.skillful_actions = ['shot', 'dribble', 'stop_dribble', 'short_pass', 'high_pass', 'long_pass']
        self.shot_bonus = 0.5
        self.pass_bonus = 0.2
        self.dribble_bonus = 0.3

    def reset(self):
        # Reset the environment and return the initial observation.
        return self.env.reset()

    def get_state(self, to_pickle):
        # Return the environment's state.
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the environment's state.
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Customize reward based on specific actions.
        observation = self.env.unwrapped.get_obs()
        components = {"base_score_reward": reward.copy(), "skill_reward": [0.0] * len(reward)}
        
        for idx in range(len(reward)):
            active_player = observation[idx]['active']
            sticky_actions = observation[idx]['sticky_actions']
            
            if 'shot' in sticky_actions and sticky_actions['shot']:
                components["skill_reward"][idx] += self.shot_bonus
            elif 'dribble' in sticky_actions and sticky_actions['dribble']:
                components["skill_reward"][idx] += self.dribble_bonus
            elif any(key in sticky_actions for key in ['short_pass', 'high_pass', 'long_pass']):
                components["skill_reward"][idx] += self.pass_bonus

            # Update the reward with the skillful action bonus
            reward[idx] += components["skill_reward"][idx]

        return reward, components

    def step(self, action):
        # Call the original step method.
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the reward() method.
        reward, components = self.reward(reward)

        # Add final reward and components information to info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
