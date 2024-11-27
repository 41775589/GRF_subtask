import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that emphasizes ball control and offensive actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialization of parameters if required
        self.ball_control_reward = 1.0
        self.goal_shot_reward = 2.0
        self sprint_bonus = 0.5

    def reset(self):
        # Reset the environment and any necessary variables
        return self.env.reset()

    def get_state(self, to_pickle):
        # Retrieve state for serialization
        state = self.env.get_state(to_pickle)
        state['ball_control_reward'] = self.ball_control_reward
        state['goal_shot_reward'] = self.goal_shot_reward
        state['sprint_bonus'] = self.sprint_bonus
        return state

    def set_state(self, state):
        # Set the environment state from a deserialized state
        self.ball_control_reward = state['ball_control_reward']
        self.goal_shot_reward = state['goal_shot_reward']
        self.sprint_bonus = state['sprint_bonus']
        return self.env.set_state(state)

    def reward(self, reward):
        # Modify reward based on the agents' interaction with the ball and sprint actions
        observations = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control_reward": [0.0] * len(reward),
            "goal_shot_reward": [0.0] * len(reward),
            "sprint_bonus": [0.0] * len(reward)
        }

        for i, observation in enumerate(observations):
            # Check if the agent is controlling the ball
            if observation['ball_owned_player'] == observation['active']:
                components['ball_control_reward'][i] = self.ball_control_reward
                reward[i] += components['ball_control_reward'][i]
            
            # Check if a goal was scored by the active agent
            if observation['score'][0] > observation['score'][1]:  # Assuming the left team is scoring
                components['goal_shot_reward'][i] = self.goal_shot_reward
                reward[i] += components['goal_shot_reward'][i]
            
            # Check if the sprint action is active for the agent
            if 'sprint' in observation['sticky_actions']:
                components['sprint_bonus'][i] = self.sprint_bonus
                reward[i] += components['sprint_bonus'][i]

        return reward, components

    def step(self, action):
        # Perform step in the environment with the given action, modify reward, and return new state
        observation, reward, done, info = self.env.step(action)
        # Calculate custom rewards
        reward, components = self.reward(reward)
        # Append the detailed reward components to info for analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
