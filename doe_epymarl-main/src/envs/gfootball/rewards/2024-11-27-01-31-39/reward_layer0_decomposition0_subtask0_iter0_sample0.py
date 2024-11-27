import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for a subtask focusing on ball control, passing, and playmaking."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for playmaking reward
        self.pass_success_reward = 0.5
        self.control_gain_reward = 0.3
        self.playmaking_reward = 0.2
        self.ball_control_reward = 0.4

    def reset(self):
        """Resets the environment for a new episode."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the state of the reward wrapper along with the environment's state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the reward wrapper along with the environment's state."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Rewards agents for effective ball control, successful passes, and good playmaking."""
        observation = self.env.unwrapped.observation()
        new_rewards = reward.copy()  # Start with the base reward
        components = {
            "base_score_reward": new_rewards,
            "pass_reward": [0.0] * len(reward),
            "control_reward": [0.0] * len(reward),
            "playmaking_reward": [0.0] * len(reward)
        }

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:  # Check if the ball is owned by the agent's team
                if obs['ball_owned_player'] == obs['active']:
                    components['control_reward'][i] = self.ball_control_reward
                if obs['game_mode'] == 0 and not np.all(obs['ball_direction'] == 0):  # Playmaking scenarios
                    components['playmaking_reward'][i] = self.playmaking_reward
            # Check for successful pass event logic (simplified assumption here)
            if obs['game_mode'] == 0 and np.linalg.norm(obs['ball_direction']) > 0.1:
                components['pass_reward'][i] = self.pass_success_reward

            # Sum up additional reward components to the base reward
            new_rewards[i] += (components['control_reward'][i] + components['playmaking_reward'][i] +
                               components['pass_reward'][i])

        return new_rewards, components

    def step(self, action):
        """Steps through the environment with the given action, applying the custom reward adjustments."""
        observation, reward, done, info = self.env.step(action)
        mod_reward, components = self.reward(reward)
        info['final_reward'] = sum(mod_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, mod_reward, done, info
