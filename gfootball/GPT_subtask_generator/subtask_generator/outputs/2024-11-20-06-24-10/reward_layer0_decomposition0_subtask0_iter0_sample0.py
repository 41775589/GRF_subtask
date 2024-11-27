import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on passing and transition skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_completion_reward = 0.5
        self.dribble_success_reward = 0.3
        self.positioning_reward = 0.2

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
            "pass_completion_reward": [0.0] * len(reward),
            "dribble_success_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 0:
                continue
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # Considering example success on dribble and positioning
                components["dribble_success_reward"][rew_index] = self.dribble_success_reward
                reward[rew_index] += components["dribble_success_reward"][rew_index]

                # Updating reward for successful positioning near goal area
                if o['right_team'][o['active']][0] > 0.5: # Example criteria
                    components["positioning_reward"][rew_index] = self.positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
