import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward to incentivize midfield control and defensive skills."""
    
    def __init__(self, env):
        super().__init__(env)
        # Good passing incentives
        self.passing_reward = 0.2
        # Interception incentives
        self.interception_reward = 0.3
        # Positional awareness reward, checking relative positioning
        self.positional_reward = 0.1

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize the components dict
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward),
            "positional_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Iterate through each agent's observations to compute additional rewards
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Handle passing reward
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if o['sticky_actions'][6] == 1:  # assuming index 6 is pass action
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += self.passing_reward

            # Interception: if the ball was owned by the other team and now it's owned by no one or our team
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                # if the ball changes from the opponent's possession to no possession
                if 'ball_owned_team' not in observation[(rew_index + 1) % len(reward)] or \
                   observation[(rew_index + 1) % len(reward)]['ball_owned_team'] != 1:
                    components["interception_reward"][rew_index] = self.interception_reward
                    reward[rew_index] += self.interception_reward

            # Positional reward: Reward based on good defensive or midfield positioning
            # Assuming good positioning means being between the ball and the goal
            if 'right_team' in o and 'ball' in o:
                own_goal = [1, 0]  # mocking a fixed position of a team’s goal post
                distance_to_goal = np.linalg.norm(np.array(own_goal) - np.array(o['right_team'][rew_index]))
                ball_distance_to_goal = np.linalg.norm(np.array(own_goal) - np.array(o['ball'][:2]))
                if ball_distance_to_goal > distance_to_goal:
                    components["positional_reward"][rew_index] = self.positional_reward
                    reward[rew_index] += self.positional_reward

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
