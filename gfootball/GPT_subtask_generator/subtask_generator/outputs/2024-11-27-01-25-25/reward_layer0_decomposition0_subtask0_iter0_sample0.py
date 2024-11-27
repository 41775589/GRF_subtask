import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds enhanced rewards focused on possession management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for possession control and pass mastery
        self.ball_control_reward = 1.0
        self.pass_accuracy_reward = 0.5
        self.dribble_skill_reward = 0.3

    def reset(self):
        """Reset the environment and any internal variables."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state of the environment with reward customizations."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from the stored state including reward customizations."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Enhance the reward function focusing on possession management actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "ball_control_reward": [], "pass_accuracy_reward": [], "dribble_skill_reward": []}
        for rew_index, o in enumerate(observation):
            # Reward for maintaining possession
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["ball_control_reward"].append(self.ball_control_reward)
                reward[rew_index] += components["ball_control_reward"][rew_index]
            else:
                components["ball_control_reward"].append(0.0)

            # Reward for successful passes
            if 'sticky_actions' in o and o['sticky_actions'][1] == 1:  # assuming action index 1 is a pass action
                components["pass_accuracy_reward"].append(self.pass_accuracy_reward)
                reward[rew_index] += components["pass_accuracy_reward"][rew_index]
            else:
                components["pass_accuracy_reward"].append(0.0)

            # Reward for dribbling skill
            if 'sticky_actions' in o and o['sticky_actions'][6] == 1:  # assuming action index 6 is dribbling
                components["dribble_skill_reward"].append(self.dribble_skill_reward)
                reward[rew_index] += components["dribble_skill_reward"][rew_index]
            else:
                components["dribble_skill_reward"].append(0.0)

        return reward, components

    def step(self, action):
        """Execute one time step within the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Traverse components dictionary to write each key-value pair into info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
