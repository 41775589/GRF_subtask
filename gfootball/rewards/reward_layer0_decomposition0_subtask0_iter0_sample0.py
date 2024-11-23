import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for ball control skills - dribbling, passing, and shooting."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for reward computation
        self.position_reward_coefficient = 1.0
        self.shot_reward_coefficient = 3.0
        self.pass_reward_coefficient = 2.0
        self.dribble_reward_coefficient = 1.5

    def reset(self):
        # Reset the environment and any necessary variables
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Initialize components for detailed reward tracking
        components = {"base_score_reward": reward.copy(),
                      "position_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for agent_idx in range(len(reward)):
            obs = observation[agent_idx]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 0:
                if 'ball_owned_player' in obs and obs['ball_owned_player'] == obs['active']:
                    # Player owns the ball, reward based on action
                    current_action = obs['sticky_actions']  # This is a simplification

                    if any(current_action[1:4]):  # Assuming these indices correspond to shot types
                        components['shot_reward'][agent_idx] = self.shot_reward_coefficient
                    if any(current_action[5:8]):  # Assuming these indices are related to passes
                        components['pass_reward'][agent_idx] = self.pass_reward_coefficient
                    if current_action[9]:  # Assuming this index is related to dribbling
                        components['dribble_reward'][agent_idx] = self.dribble_reward_coefficient

                    # Position-based reward, the closer to goal, the higher the reward
                    goal_position = [1, 0]
                    distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(obs['ball']))
                    position_reward = self.position_reward_coefficient / (distance_to_goal + 1e-4)
                    components['position_reward'][agent_idx] = position_reward

                    # Aggregate rewards
                    reward[agent_idx] += sum(components[key][agent_idx] for key in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add each reward component to info for debugging or further analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
