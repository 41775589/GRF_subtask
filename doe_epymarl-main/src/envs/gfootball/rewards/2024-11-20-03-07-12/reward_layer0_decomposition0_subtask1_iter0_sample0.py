import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored reward for defensive and midfield control skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'game_mode' not in o:
                continue

            # Encourage maintaining possession: reward for keeping the ball longer
            if o['ball_owned_team'] == 1:  # Assuming the agent's team index is 1
                components["defensive_reward"][rew_index] += 0.1  # Reward for possession
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Reward successful passes between players
            if o['game_mode'] == 1 or o['game_mode'] == 2:  # If the game mode indicates a free kick or throw-in
                components["defensive_reward"][rew_index] += 0.5
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Reward interceptions: if ball possession changes unfavorably, penalize slightly
            previous_owned = self.env.unwrapped.get_state({}).get('prev_ball_owned_team', -1)
            if previous_owned != -1 and previous_owned != o['ball_owned_team'] and o['ball_owned_team'] == 0:
                components["defensive_reward"][rew_index] -= 0.3
                reward[rew_index] += components["defensive_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
