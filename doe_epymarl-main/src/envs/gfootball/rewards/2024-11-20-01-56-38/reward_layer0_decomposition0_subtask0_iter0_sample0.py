import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive maneuvers for training in football."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters to tune effectiveness of different actions
        self.goal_distance_reward_coefficient = 1.0
        self.shot_reward_coefficient = 5.0
        self.pass_reward_coefficient = 3.0
        self.dribble_reward_coefficient = 2.0

    def reset(self):
        """Resets the environment and necessary variables."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Helper function to get the current state with user-defined alterations."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the current state from a saved state, ensuring to include wrapper modifications."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modifies the rewards returned by the environment based on offensive subtask objectives."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "goal_distance_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward), "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                # Calculate distance to goal (normalized to range 0-1, goal at y=0)
                goal_distance = 1 - np.abs(o['ball'][1])
                components["goal_distance_reward"][rew_index] = self.goal_distance_reward_coefficient * goal_distance
                
                # Check which action was taken and assign extra rewards for dribbling closer to goal, making passes and shots
                last_action = o['sticky_actions'][-1]  # Accessing the last action which could be the most relevant
                if last_action in [1, 6]:  # Ids for shot actions (assuming id 1 and 6 refers to different shot types)
                    components["shot_reward"][rew_index] = self.shot_reward_coefficient
                if last_action in [2, 3, 4]:  # Ids for pass actions (short, long, high passes assumed)
                    components["pass_reward"][rew_index] = self.pass_reward_coefficient
                if last_action in [9]:  # Id for dribble action
                    components["dribble_reward"][rew_index] = self.dribble_reward_coefficient * goal_distance

            # Aggregate the rewards
            reward[rew_index] = (components["base_score_reward"][rew_index] +
                                 components["goal_distance_reward"][rew_index] +
                                 components["shot_reward"][rew_index] +
                                 components["pass_reward"][rew_index] +
                                 components["dribble_reward"][rew_index])

        return reward, components

    def step(self, action):
        """Steps through the environment, modifying the reward using the reward() method."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Add each reward component to the info for potential debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
