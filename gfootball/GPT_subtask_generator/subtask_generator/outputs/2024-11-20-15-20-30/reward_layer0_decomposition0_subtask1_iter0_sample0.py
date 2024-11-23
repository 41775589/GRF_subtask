import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a positional and defensive action-related reward."""

    def __init__(self, env):
        super().__init__(env)
        self.positional_reward_scale = 0.1
        self.defensive_action_reward_scale = 0.2

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "defensive_action_reward": [0.0] * len(reward)}
        
        for rew_index, o in enumerate(observation):
            # Encourage moving closer to the ball
            if 'ball' in o:
                ball_dist = np.linalg.norm(o['left_team'][o['active']] - o['ball'][:2])
                # positional reward for being close to the ball
                components["positional_reward"][rew_index] = (1 - ball_dist) * self.positional_reward_scale
                reward[rew_index] += components["positional_reward"][rew_index]

            # Reward for defensive actions
            if 'sticky_actions' in o:
                # Defensive actions: sliding (index 8), sprint (index 9) - indices based on action set
                if o['sticky_actions'][8] == 1 or o['sticky_actions'][9] == 1:
                    components["defensive_action_reward"][rew_index] = self.defensive_action_reward_scale
                    reward[rew_index] += components["defensive_action_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
