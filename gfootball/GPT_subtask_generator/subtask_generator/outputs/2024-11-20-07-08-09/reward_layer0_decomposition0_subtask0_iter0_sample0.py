import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for offensive strategy training."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_control_score = 0.3
        self.shot_score = 0.5
        self.dribbling_score = 0.2

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Performance states can be restored here if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            player_has_ball = (obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active'])

            if player_has_ball:
                # Enhance rewards around ball control
                components["ball_control_reward"][rew_index] += self.ball_control_score
                
                # If the player shoot towards net
                if obs['sticky_actions'][7] == 1:
                    components["shot_reward"][rew_index] += self.shot_score

                # Reward dribbling by action index - assuming action index 'x' relates to dribbling
                if obs['sticky_actions'][5] == 1:  # Assuming action index 5 is dribble
                    components["dribbling_reward"][rew_index] += self.dribbling_score

            # Combine all rewards and components
            total_reward = (components["base_score_reward"][rew_index] +
                            components["ball_control_reward"][rew_index] +
                            components["shot_reward"][rew_index] +
                            components["dribbling_reward"][rew_index])
            reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
