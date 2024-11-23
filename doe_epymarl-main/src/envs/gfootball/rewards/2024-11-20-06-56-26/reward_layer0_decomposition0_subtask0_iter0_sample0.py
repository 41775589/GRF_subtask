import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function specifically designed for offensive football plays."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        
        self.shooting_range_threshold = 0.3  # Threshold to consider the ball is close enough to shoot effectively
        self.dribble_controls_required = 5   # Number of dribbles required to maximally reward dribbling skill improvement
        self.passing_accuracy_threshold = 0.2  # Threshold for precision in passing, close enough to a teammate
        self.pass_reward_coefficient = 0.3
        self.shot_reward_coefficient = 0.5
        self.dribble_reward_coefficient = 0.2

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
        components = {"base_score_reward": reward.copy(),
                      "shot_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:  # Assuming the team with index 0 is the controlled team
                ball_pos = obs['ball']
                # Calculate the distance from goal
                distance_to_goal = np.linalg.norm(ball_pos[0:2] - np.array([1, 0]))  # Goal at (1,0)

                # Check if it's a good moment to shoot
                if distance_to_goal < self.shooting_range_threshold:
                    components["shot_reward"][index] = self.shot_reward_coefficient
                    reward[index] += components["shot_reward"][index]

                # Check for successful passes
                if obs['sticky_actions'][1] == 1 or obs['sticky_actions'][2] == 1:  # Indices for pass actions
                    components["pass_reward"][index] = self.pass_reward_coefficient
                    reward[index] += components["pass_reward"][index]

                # Encourage dribbling by checking dribble action frequency
                dribble_action = obs['sticky_actions'][3]  # Index for dribble action
                if dribble_action:
                    components["dribble_reward"][index] = self.dribble_reward_coefficient
                    reward[index] += components["dribble_reward"][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
