import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies reward function to emphasize defensive coordination and positioning in football game."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Adjusting these parameters to better control the reward scale.
        self.defensive_action_bonus = 0.3
        self.positioning_bonus_scale = 0.2
        self.sprint_efficiency_bonus = 0.1
        self.base_score_scale = 1.0

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": [r * self.base_score_scale for r in reward],
            "defensive_action_bonus": [0.0] * len(reward),
            "sprint_efficiency_bonus": [0.0] * len(reward),
            "positioning_bonus": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            obs = observation[i]

            # Defensive Action Bonus: Increase reward if the player performs a slide (assuming index for slide=7)
            if obs['sticky_actions'][7] == 1:
                components['defensive_action_bonus'][i] = self.defensive_action_bonus

            # Sprint Efficiency Bonus: Reward using sprint effectively (index for sprint=1)
            if obs['sticky_actions'][1] == 1:
                components['sprint_efficiency_bonus'][i] = self.sprint_efficiency_bonus

            # Positioning Bonus: Reinforce if the player is close to defending key field positions
            player_position = obs['left_team'][obs['active']]
            key_position = [0.1, -0.1]  # Example coordinates for a strategic defensive position
            distance = np.linalg.norm(player_position - key_position)
            if distance < 0.1:
                components['positioning_bonus'][i] = self.positioning_bonus_scale

            reward[i] = sum([
                components['base_score_reward'][i],
                components['defensive_action_bonus'][i],
                components['sprint_efficiency_bonus'][i],
                components['positioning_bonus'][i]
            ])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
