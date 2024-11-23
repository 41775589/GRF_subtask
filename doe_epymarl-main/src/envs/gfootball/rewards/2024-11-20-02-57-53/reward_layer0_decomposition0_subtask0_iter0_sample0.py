import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward for offensive actions such as successful dribbling, 
    shooting accuracy, and effective passing as part of a sub-task curriculum."""

    def __init__(self, env):
        super().__init__(env)
        self.goal_distance_reward_coefficient = 0.1
        self.successful_pass_reward = 0.2
        self.shot_on_target_reward = 0.5
        self.successful_dribble_reward = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goal_distance_reward": [0.0] * len(reward),
            "successful_pass_reward": [0.0] * len(reward),
            "shot_on_target_reward": [0.0] * len(reward),
            "successful_dribble_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Enhancing reward based on proximity to the opponent's goal when possessing the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                distance_to_goal = np.linalg.norm(o['ball'][:2] - np.array([1, 0]))  # assuming goal at (1,0)
                components["goal_distance_reward"][rew_index] = (1 - distance_to_goal) * self.goal_distance_reward_coefficient
                reward[rew_index] += components["goal_distance_reward"][rew_index]

            # Reward for successful passes
            if 'successful_pass' in o['sticky_actions']:
                components["successful_pass_reward"][rew_index] = self.successful_pass_reward
                reward[rew_index] += components["successful_pass_reward"][rew_index]

            # Reward for shots on target
            if 'shot_on_target' in o['sticky_actions']:
                components["shot_on_target_reward"][rew_index] = self.shot_on_target_reward
                reward[rew_index] += components["shot_on_target_reward"][rew_index]

            # Reward for successful dribble
            if 'successful_dribble' in o['sticky_actions']:
                components["successful_dribble_reward"][rew_index] = self.successful_dribble_reward
                reward[rew_index] += components["successful_dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
