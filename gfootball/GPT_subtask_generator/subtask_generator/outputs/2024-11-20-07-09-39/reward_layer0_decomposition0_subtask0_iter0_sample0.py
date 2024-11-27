import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to enhance training focus on dribbling,
    maintaining possession under pressure, and precise close-range shots.
    """
    def __init__(self, env):
        super().__init__(env)
        self.dribble_reward = 0.05
        self.possession_reward = 0.1
        self.shot_accuracy_reward = 0.2

    def reset(self):
        # Reset the environment state and any necessary variables
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the state of the environment for checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the state of the environment from checkpoints
        return self.env.set_state(state)

    def reward(self, reward):
        # Modify the reward based on custom conditions
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward),
            "shot_accuracy_reward": [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                # Reward for maintaining possession
                components["possession_reward"][rew_index] = self.possession_reward
                reward[rew_index] += components["possession_reward"][rew_index]

                if 'sticky_actions' in o:
                    # Check if dribble action is active
                    if o['sticky_actions'][6] == 1:  # assuming index 6 is dribbling
                        components["dribble_reward"][rew_index] = self.dribble_reward
                        reward[rew_index] += components["dribble_reward"][rew_index]

            # Reward for shot accuracy based on proximity to the goal and actual shot taken
            if 'shot' in o and np.linalg.norm(o['ball'][:2] - np.array([1, 0])) < 0.1:
                components["shot_accuracy_reward"][rew_index] = self.shot_accuracy_reward
                reward[rew_index] += components["shot_accuracy_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Obtain the results from the environment's step function
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the custom reward function
        reward, components = self.reward(reward)
        # Add final reward to the info for tracking
        info["final_reward"] = sum(reward)

        # Add all reward components to info for detailed analysis/tracking
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
