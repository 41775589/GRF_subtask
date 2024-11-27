import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that adds rewards based on dribbling and shooting close to the goal."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Defining five zones approaching the opponent's goal
        self._zone_bonus = 0.2  # Bonus reward for advancing the ball into a new closer zone

    def reset(self):
        self._last_zone = [0] * 2  # Initialize last zone for each agent
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._last_zone
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_zone = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy(), 'zone_bonus_reward': [0.0] * len(reward)}

        components = {
            "base_score_reward": reward.copy(),
            "zone_bonus_reward": [0.0] * len(reward)
        }

        # Calculate rewards based on ball position and player actions related to dribbling or shooting
        for i in range(len(reward)):
            o = observation[i]
            ball_x = o['ball'][0]  # Simplistic assumption: The goal is at x=1
            current_zone = min(int(ball_x * self._num_zones), self._num_zones - 1)

            # Check if the agent with ball has moved closer to the goal since the last step
            if current_zone > self._last_zone[i] and o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                reward[i] += self._zone_bonus
                components["zone_bonus_reward"][i] = self._zone_bonus

            self._last_zone[i] = current_zone

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
