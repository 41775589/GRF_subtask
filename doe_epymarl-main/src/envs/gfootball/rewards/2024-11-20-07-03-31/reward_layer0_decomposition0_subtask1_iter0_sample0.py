import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds custom rewards for scoring and defensive actions specific to the subtask curriculum."""

    def __init__(self, env):
        super().__init__(env)
        # Custom internal state initialization if necessary can be placed here

    def reset(self):
        # Resets any necessary internal data at the start of an episode
        return self.env.reset()

    def get_state(self, to_pickle):
        # Optionally implement state getter if required for serialization
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Optionally implement state setter if required for deserialization
        return self.env.set_state(state)

    def reward(self, reward):
        # Implement custom logic to adjust the reward based on the observations from the environment
        observation = self.env.unwrapped.observation()
        if not observation:
            return reward, {}

        # Initialize the reward components dictionary
        components = {"base_score_reward": reward.copy(), "scoring_reward": [0.0] * 3, "defensive_reward": [0.0] * 3}

        for i, rew in enumerate(reward):
            obs = observation[i]
            
            # Defining rewards for successful defense
            if obs['ball_owned_team'] == 1:  # If the opposition team owns the ball
                components['defensive_reward'][i] = 0.1  # Encourage to get ball back
            
            # Defining rewards for successful offensive actions
            if obs['ball_owned_team'] == 0:  # If our team owns the ball
                if obs['sticky_actions'][7] == 1:  # Agent made a shot action
                    components['scoring_reward'][i] = 0.3  # Encourage shooting when possible

            # Aggregating rewards modifying the original list
            reward[i] += components['scoring_reward'][i] + components['defensive_reward'][i]

        return reward, components

    def step(self, action):
        # Perform step using provided action
        observation, reward, done, info = self.env.step(action)
        
        # Modify the reward with custom logic via the reward method
        reward, components = self.reward(reward)
        
        # Add detailed reward components for analysis in the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
