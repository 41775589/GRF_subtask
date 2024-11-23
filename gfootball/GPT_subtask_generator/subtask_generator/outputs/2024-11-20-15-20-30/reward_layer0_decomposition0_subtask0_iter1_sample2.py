import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom wrapper to add enhanced reward feedback focused on ball control and plays."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward = 2.0  # Reward multiplier for successful shots
        self.passing_reward = 1.5   # Reward for successful passes
        self.dribbling_reward = 1.0 # Reward for dribbling
        self.positioning_reward = 0.5 # Positional advantage towards opponent's goal

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward)
        }

        if observation:
            for idx, obs in enumerate(observation):
                if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                    # Calculate rewards based on player actions and positions
                    components['shot_reward'][idx] += self.shooting_reward if 'Shot' in obs['sticky_actions'] else 0
                    components['pass_reward'][idx] += self.passing_reward if 'Pass' in obs['sticky_actions'] else 0
                    components['dribble_reward'][idx] += self.dribbling_reward if 'Dribble' in obs['sticky_actions'] else 0
                    
                    # Position reward - compute distance to opponent's goal and invert it as reward
                    opponent_goal_pos = [1, 0]  # Assuming this to be the coordinate of opponent's goal
                    distance_to_goal = np.linalg.norm(np.array(opponent_goal_pos) - np.array(obs['ball'][0:2]))
                    components['position_reward'][idx] += self.positioning_reward / (distance_to_goal + 0.1)

                    # Aggregate rewards
                    reward[idx] = sum(components[key][idx] for key in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Add each reward component to info for detailed feedback
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
