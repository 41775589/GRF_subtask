import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized offensive strategy reward for mastering 'Shot', 'Dribble', and 'Sprint'."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize additional metadata for tracking progress in offensive strategies
        self.shot_threshold = 0.15  # Threshold distance to goal to count as a high-reward shot opportunity
        self.dribble_award = 0.2    # Additional reward for successfully dribbling closer to the goal
        self.sprint_bonus = 0.1     # Bonus for sprinting effectively towards the opponent's half
    
    def reset(self):
        # Reset the environment and any custom data
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include state of the current wrapper
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any custom state if necessary
        return from_pickle

    def reward(self, reward):
        # Overriding the reward function to focus on offensive strategies
        observation = self.env.unwrapped.observation()[0]  # Assuming single environment for clarity
        components = {
            "base_score_reward": np.array(reward),
            "shot_reward": 0.0,
            "dribble_reward": 0.0,
            "sprint_reward": 0.0
        }

        # Check for shooting opportunity and position
        if observation['ball_owned_team'] == 0 and np.linalg.norm(observation['ball'][:2]) < self.shot_threshold:
            components['shot_reward'] += 1.0
        
        # Dribbling closer to the goal area
        if observation['ball_owned_team'] == 0 and 'Dribble' in observation['sticky_actions']:
            components['dribble_reward'] += self.dribble_award

        # Using sprint effectively toward the opponent's half
        if observation['ball_owned_team'] == 0 and 'Sprint' in observation['sticky_actions'] and observation['ball'][0] > 0:
            components['sprint_reward'] += self.sprint_bonus
        
        # Combine the components into final reward
        final_rewards = np.array([components['base_score_reward'][0] + 
                                  components['shot_reward'] +
                                  components['dribble_reward'] +
                                  components['sprint_reward']])
        
        return final_rewards, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)
        # Attach component values to info
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
