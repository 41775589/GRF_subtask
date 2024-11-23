import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense tactical reward for defense and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.intercept_position_reward = 0.2  # Reward given for moving to strategic positions
        self.defensive_actions_reward = 0.5   # Reward for engaging in defensive actions (e.g., Sliding)
        self.sprint_bonus = 0.3              # Bonus reward for utilizing sprint effectively
        self.position_threshold = 0.1        # Threshold to decide if agent is in a strategic position

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        original_reward = reward.copy()
        components = {
            "base_score_reward": original_reward,
            "tactical_positioning_reward": [0.0] * len(reward),
            "defensive_action_bonus": [0.0] * len(reward),
            "sprint_bonus": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for index in range(len(reward)):
            agent_obs = observation[index]
            # Calculate distance to strategic positions (hypothetical values)
            dist_to_position = np.linalg.norm(agent_obs['left_team'] - np.array([0.5, 0.0]))
            if dist_to_position < self.position_threshold:
                components['tactical_positioning_reward'][index] = self.intercept_position_reward

            # Check for defensive actions, e.g., if the 'Sliding' action was taken
            if agent_obs['sticky_actions'][7] == 1:  # assuming index 7 is 'Sliding'
                components['defensive_action_bonus'][index] = self.defensive_actions_reward

            # Reward for sprint usage in defensive scenarios
            if agent_obs['sticky_actions'][1] == 1:  # assuming index 1 is 'Sprint'
                components['sprint_bonus'][index] = self.sprint_bonus

            # Aggregate all components into the final reward for this agent
            reward[index] = sum([
                original_reward[index],
                components['tactical_positioning_reward'][index],
                components['defensive_action_bonus'][index],
                components['sprint_bonus'][index]
            ])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Add final reward and individual component values to the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
