import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for specific actions leading to goal-scoring opportunities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)

    def reset(self):
        """Reset environment and necessary variables."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment, with any modifications needed from this wrapper."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment, with any modifications needed from this wrapper."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Reward function to encourage actions that contribute to creating and utilizing goal-scoring opportunities.
        
        The reward considers actions like shooting, high passes (potential assists), and sprints (positioning).
        """
        observation = self.env.unwrapped.observation()
        
        # Build components dictionary to store different parts of the reward
        components = {"base_score_reward": reward.copy(), "goal_oriented_actions_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for shot attempts, penalize or reward based on proximity to goal
            if o['action'] == 'shot':
                # Let's assume proximities could be measured or inferred from location data
                distance_to_goal = np.linalg.norm(np.array(o['ball']) - np.array([1, 0]))  # Estimate distance to opponent's goal at (1,0)
                if distance_to_goal < 0.2:  # Arbitrary threshold for "good shot range"
                    components["goal_oriented_actions_bonus"][rew_index] = 1.0
                else:
                    components["goal_oriented_actions_bonus"][rew_index] = -0.5

            # Check for high passes, simple bonus for successful high pass
            if o['action'] == 'high_pass':
                components["goal_oriented_actions_bonus"][rew_index] = 0.3

            # Check for sprints, which should ideally position player closer to the opponent's goal
            if o['action'] == 'sprint':
                components["goal_oriented_actions_bonus"][rew_index] = 0.1

            # Update overall reward with the bonuses and penalties
            reward[rew_index] += components["goal_oriented_actions_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """
        Overridden step function to include custom reward adjustments.
        
        Calls the original step method, modifies the reward, and returns modified observation and reward.
        """
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the overridden reward function
        reward, components = self.reward(reward)

        # Add final reward and reward components to the info dictionary for tracking and debugging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
