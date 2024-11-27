import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective ball control and shooting actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribble_reward = 0.3
        self.shot_reward = 0.5
        self.positioning_reward = 0.2

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['reward_state'] = "Custom state information can be added here if needed"
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        from_pickle['reward_state'] = "Custom state setting can be loaded here if needed"
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_rewards = [0.0] * len(reward)
        components = {'base_score_reward': reward.copy(), 
                      'dribble_reward': [0.0] * len(reward),
                      'shot_reward': [0.0] * len(reward),
                      'positioning_reward': [0.0] * len(reward)}
        
        for agent_index in range(len(reward)):
            o = observation[agent_index]

            # Check if the agent has the ball
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                if 'dribble' in o['sticky_actions']:
                    # Reward for dribbling
                    new_rewards[agent_index] += self.dribble_reward
                    components['dribble_reward'][agent_index] = self.dribble_reward

                if o['game_mode'] == 0 and np.linalg.norm(o['ball'] - np.array([1, 0, 0])) < 0.1:
                    # Reward for shots near the goal
                    new_rewards[agent_index] += self.shot_reward
                    components['shot_reward'][agent_index] = self.shot_reward

            # Positioning reward when not in possession to get in better positions
            if o['ball_owned_team'] != 0:
                new_rewards[agent_index] += self.positioning_reward
                components['positioning_reward'][agent_index] = self.positioning_reward

        # Accumulate rewards and update observations with new rewards
        reward = [original + add for original, add in zip(reward, new_rewards)]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
