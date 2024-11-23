import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper designed to enhance reward signals for defensive and tactical strategic training in a football environment."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.position_importance = 1.0       # Increased importance for tactical position
        self.defense_importance = 1.0        # Emphasizing defense actions
        self.sprint_importance = 0.1         # Decreased importance for sprint bonus due to low variance in learning signal
        self.positioning_threshold = 0.2     # Threshold for rewarding tactical positioning

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
        tactical_positioning_rewards = [0.0] * len(reward)
        defensive_action_rewards = [0.0] * len(reward)
        sprint_rewards = [0.0] * len(reward)

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        for index in range(len(reward)):
            agent_obs = observation[index]

            # Reward for positioning closer to key defensive areas, which is center back positions
            if 'right_team' in agent_obs:
                player_pos = agent_obs['right_team'][agent_obs['active']]
                # Simple model to estimate key defensive region (usually central lower-third)
                if player_pos[0] < 0.33 and abs(player_pos[1]) < 0.25:
                    tactical_positioning_rewards[index] = self.position_importance * self.positioning_threshold / (np.linalg.norm([0.0, 0.0] - player_pos) + 0.1)

            # Reward defensive actions: check if 'Sliding' action is taken
            if agent_obs['sticky_actions'][7] == 1:  # Assuming index 7 represents 'Sliding'
                defensive_action_rewards[index] = self.defense_importance * 0.5

            # Small reward for sprint utilization if it optimizes defensive reaction
            if agent_obs['sticky_actions'][6] == 1:  # Assuming index 6 represents 'Sprint'
                sprint_rewards[index] = self.sprint_importance * 0.2

            # Consolidating the rewards into final reward
            reward[index] += (tactical_positioning_rewards[index] +
                              defensive_action_rewards[index] +
                              sprint_rewards[index])

        components = {
            'base_score_reward': original_reward,
            'tactical_positioning_reward': tactical_positioning_rewards,
            'defensive_action_bonus': defensive_action_rewards,
            'sprint_bonus': sprint_rewards
        }

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add final reward and split component values to info for better diagnostics and learning analytics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
