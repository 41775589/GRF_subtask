import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive skills including dribbling, shooting, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.dribble_weight = 0.8
        self.shot_weight = 1.0
        self.pass_weight = 0.5
        self.positional_weight = 0.2

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
            "dribble_reward": [0.0] * len(reward),
            "shot_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "positional_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Ensure the length of the list is as expected
        assert len(reward) == 2

        for index in range(len(reward)):
            player_obs = observation[index]

            if player_obs['ball_owned_player'] == player_obs['active']:
                # Influence reward based in possession and action type
                active_actions = player_obs['sticky_actions']
                dribbling = active_actions[4]  # assuming index 4 corresponds to dribbling
                shooting = active_actions[9]  # assuming index 9 corresponds to shooting
                passing = active_actions[2] or active_actions[3]  # short or long pass

                # Positional reward for getting closer to the opponent's goal
                x_position = player_obs['left_team'][player_obs['active']][0]
                score_zone_bonus = max(0, x_position - 0.5)  # only reward forward of the mid-line

                # Calculate the individual rewards
                if dribbling:
                    components['dribble_reward'][index] = self.dribble_weight
                if shooting:
                    components['shot_reward'][index] = self.shot_weight
                if passing:
                    components['pass_reward'][index] = self.pass_weight
                components['positional_reward'][index] = self.positional_weight * score_zone_bonus

                # Update the base reward
                reward[index] += (components['dribble_reward'][index] +
                                  components['shot_reward'][index] +
                                  components['pass_reward'][index] +
                                  components['positional_reward'][index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
