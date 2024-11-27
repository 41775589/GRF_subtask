import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific dense reward focused on attacking skills."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_reward = 0.2
        self.pass_reward = 0.1
        self.shoot_reward = 0.5
        self.goal_reward = 1.0

    def reset(self):
        """Reset the environment and clear any internal state."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state with possible checkpoint reward information."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state and update internal checkpoint reward states."""
        from_pickle = self.env.set_state(state)
        # Not specifically using checkpoints yet, just an example
        return from_pickle

    def reward(self, reward):
        """Calculate additional rewards focusing on the attacking skills subtask."""
        observation = self.env.unwrapped.observation()
        reward_modified = reward.copy()

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for dribbling closer to the opponent's goal
            goal_distance = np.linalg.norm(o['ball'] - np.array([1, 0]))
            if o['ball_owned_player'] == o['active']:
                reward_modified[rew_index] += self.dribble_reward / goal_distance

            # Reward for successful passes in the attacking half
            if o['ball_owned_team'] == 0 and o['ball'][0] > 0:
                reward_modified[rew_index] += self.pass_reward

            # Additional reward if the shot leads to a goal
            if o['game_mode'] == 0 and goal_distance < 0.1:  # Assuming game_mode 0 is a shooting opportunity
                reward_modified[rew_index] += self.shoot_reward
            if o['score'][0] > 0:  # Assuming score for team 0 increases due to a goal
                reward_modified[rew_index] += self.goal_reward

            # Ensuring only positive rewards
            reward_modified[rew_index] = max(reward_modified[rew_index], 0)

        components = {
            "base_score_reward": reward,
            "dribble_reward": [self.dribble_reward if r > 0 else 0 for r in reward_modified],
            "pass_reward": [self.pass_reward if r > 0 else 0 for r in reward_modified],
            "shoot_reward": [self.shoot_reward if r > 0 else 0 for r in reward_modified],
            "goal_reward": [self.goal_reward if r > 0 else 0 for r in reward_modified]
        }

        return reward_modified, components

    def step(self, action):
        """Step the environment, adjust the reward, and return modified information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)

        # Store the reward components for detailed analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
