import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for controlling game tempo and maintaining possession."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_accuracy_reward = 0.2  # Reward for successful passes.
        self.possession_reward = 0.1      # Reward for maintaining possession.
        self.movement_reward = 0.05       # Reward for effective movement.

    def reset(self):
        """Reset the environment and clear any internal states."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper from serialized data."""
        return self.env.set_state(state)
    
    def reward(self, reward):
        """Modify the rewards given by the environment based on possession and passing metrics."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward),
                      "possession_reward": [0.0] * len(reward),
                      "movement_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for idx, rewards in enumerate(reward):
            obs = observation[idx]
            
            # Encourage passing by rewarding successful passes
            if obs['game_mode'] in [2, 3, 4]:  # These game modes are related to passing outcomes
                components["pass_accuracy_reward"][idx] += self.pass_accuracy_reward
            
            # Encourage keeping possession of the ball
            if obs['ball_owned_team'] == obs['active']:
                components["possession_reward"][idx] += self.possession_reward
            
            # Reward effective movement in purposeful directions
            player_pos_before = np.array(obs['left_team'] if obs['active'] else obs['right_team'])
            player_pos_after = player_pos_before + np.array(obs['left_team_direction'] if obs['active'] else obs['right_team_direction'])
            if np.any(player_pos_after - player_pos_before != 0):
                components["movement_reward"][idx] += self.movement_reward
            
            # Aggregate custom rewards with the base game reward
            reward[idx] += sum(components[k][idx] for k in components.keys())

        return reward, components

    def step(self, action):
        """Take a step using the given actions, modify the reward, and return the results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add final reward to the info for diagnostic purposes
        info["final_reward"] = sum(reward)

        # Add each component of the reward to the info for diagnostics
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
