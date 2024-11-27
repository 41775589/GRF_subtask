import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering possession and passing skills."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_accuracy_reward = 0.5

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
                      "pass_accuracy_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Enhance rewards for successful passes
            is_pass = o['sticky_actions'][5] or o['sticky_actions'][6] or o['sticky_actions'][7]  # Assuming indices for short, high, long pass
            if is_pass and o['ball_owned_team'] == 0:  # Assuming team 0 is the controlled team
                if o['ball_owned_player'] == o['active']:
                    components["pass_accuracy_reward"][rew_index] = self.pass_accuracy_reward
                    reward[rew_index] += components["pass_accuracy_reward"][rew_index]

            # Consider movement rewards
            if np.any(o['left_team_direction'][rew_index] != 0) or np.any(o['right_team_direction'][rew_index] != 0):
                components["pass_accuracy_reward"][rew_index] += 0.1  # Reward for movement
                reward[rew_index] += 0.1  # Update the reward list directly
        
        return reward, components

    def step(self, action):
        # Call the original step method which returns the observation, reward, done, info
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the customized reward() method
        reward, components = self.reward(reward)

        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
