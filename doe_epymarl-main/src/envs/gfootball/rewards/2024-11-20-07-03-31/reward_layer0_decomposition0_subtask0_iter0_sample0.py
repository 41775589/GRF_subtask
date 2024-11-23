import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a passing and movement-focused reward for managing gameplay and maintaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self._num_pass_attempts = 0
        self._successful_passes = 0
        self._movement_precision = 0.1  # The reward increment for player's strategic repositioning

    def reset(self):
        self._num_pass_attempts = 0
        self._successful_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'num_pass_attempts': self._num_pass_attempts,
            'successful_passes': self._successful_passes
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_pass_attempts = from_pickle['CheckpointRewardWrapper']['num_pass_attempts']
        self._successful_passes = from_pickle['CheckpointRewardWrapper']['successful_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "movement_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Add rewards for passing
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                self._num_pass_attempts += 1
                if np.random.rand() < 0.7:  # Simulate a 70% chance of successful pass
                    self._successful_passes += 1
                    components['passing_reward'][i] = max(1, (self._successful_passes / self._num_pass_attempts))
                    reward[i] += components['passing_reward'][i]

            # Add rewards for strategic movement
            if 'left_team_direction' in o:
                move_quality = 0.05  # Every move increases reward slightly if in correct direction
                if np.linalg.norm(o['left_team_direction'][o['active']]) > 0:
                    direction = o['left_team_direction'][o['active']]/np.linalg.norm(o['left_team_direction'][o['active']])
                    target_direction = np.array([1, 0])  # Hypothetical optimal direction towards goal
                    alignment = np.dot(direction, target_direction)
                    components['movement_reward'][i] = move_quality * alignment
                    reward[i] += components['movement_reward'][i]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info for tracking
        info["final_reward"] = sum(reward)
        # Traverse the components dictionary and write each key-value pair into info with the prefix 'component_'
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
