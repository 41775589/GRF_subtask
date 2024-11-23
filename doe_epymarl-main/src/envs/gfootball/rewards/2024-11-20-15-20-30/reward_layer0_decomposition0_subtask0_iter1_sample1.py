import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the base reward to emphasize individual technical skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Coefficients for different types of individual actions
        self.shoot_coefficient = 2.0
        self.dribble_coefficient = 1.5
        self.pass_coefficient = 2.0
        self.position_coefficient = 1.0

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        modified_reward = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "shoot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward)
        }

        observation = self.env.unwrapped.observation()
        if observation is None:
            return modified_reward, components

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:  # Assuming team 0 is controlled
                # Encourage shooting towards goal
                if any(action in obs['sticky_actions'][1:4]):  # Presumed indices for shooting actions
                    components['shoot_reward'][i] = self.shoot_coefficient
                # Encourage dribbling
                if obs['sticky_actions'][9]:  # Presumed index for dribbling
                    components['dribble_reward'][i] = self.dribble_coefficient
                # Encourage passing
                if any(action in obs['sticky_actions'][5:8]):  # Presumed indices for passing
                    components['pass_reward'][i] = self.pass_coefficient
                # Positional reward - closer to the opponent's goal is better
                goal_position = np.array([1, 0])  # Hypothetical opponent goal position on normalized field
                ball_position = np.array(obs['ball'][:2])  # Ignore z-coordinate in position
                distance_to_goal = np.linalg.norm(goal_position - ball_position)
                components['position_reward'][i] = self.position_coefficient / (distance_to_goal + 0.1)  # Avoid division by zero
                
                # Aggregate additional rewards
                total_additional_reward = (components['shoot_reward'][i] +
                                           components['dribble_reward'][i] +
                                           components['pass_reward'][i] +
                                           components['position_reward'][i])
                modified_reward[i] += total_additional_reward

        return modified_reward, components

    def step(self, action):
        observation, base_reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(base_reward)
        
        # Packaging the final reward and components into info for debugging
        info["final_reward"] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, modified_reward, done, info
