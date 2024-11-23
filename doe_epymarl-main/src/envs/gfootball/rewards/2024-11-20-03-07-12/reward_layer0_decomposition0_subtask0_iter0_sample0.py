import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive tactics with ball control and shot execution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_target_bonus = 0.1  # Bonus for moving the ball towards opponent's goal
        self.shot_on_goal_bonus = 0.3  # Bonus for shots towards goal
        self.dribble_bonus = 0.05  # Bonus for successful dribbles

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control_bonus": [0.0] * len(reward),
            "goal_shot_bonus": [0.0] * len(reward),
            "dribble_bonus": [0.0] * len(reward)
        }

        if obs is None:
            return reward, components

        for i in range(len(reward)):
            o = obs[i]

            # Bonus for ball control: move the ball towards the opponent's side
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                opponents_goal = 1.0  # Assuming opponents' goal is always at x=1
                ball_position_x = o['ball'][0]
                control_bonus = (opponents_goal - ball_position_x) * self.ball_target_bonus
                components["ball_control_bonus"][i] = control_bonus
                reward[i] += control_bonus

            # Goal shot bonus
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if o['sticky_actions'][5] == 1:  # Assuming index 5 is the "shot" action
                    components["goal_shot_bonus"][i] = self.shot_on_goal_bonus
                    reward[i] += self.shot_on_goal_bonus

            # Dribbling bonus
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if o['sticky_actions'][7] == 1:  # Assuming index 7 is the "dribble" action
                    components["dribble_bonus"][i] = self.dribble_bonus
                    reward[i] += self.dribble_bonus

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse through the components dictionary and add each key-value pair to info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
