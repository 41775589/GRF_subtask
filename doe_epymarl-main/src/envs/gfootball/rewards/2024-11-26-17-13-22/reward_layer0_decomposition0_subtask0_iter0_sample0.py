import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward based on offensive strategies for shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define some internal thresholds and multipliers for reward calculation
        self.shot_accuracy_threshold = 0.2
        self.dribble_efficiency_threshold = 0.3
        self.pass_accuracy_threshold = 0.5
        self.shot_reward_multiplier = 2.0
        self.dribble_reward_multiplier = 1.5
        self.pass_reward_multiplier = 1.0

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Access the observation from the unwrapped environment directly
        observation = self.env.unwrapped.observation()

        # Adjust rewards based on custom strategy-focused goals
        new_rewards = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "shot_accuracy_reward": [0.0] * 2,
            "dribble_efficiency_reward": [0.0] * 2,
            "pass_accuracy_reward": [0.0] * 2,
        }
        
        for idx, agent_obs in enumerate(observation):
            if agent_obs['game_mode'] == 0:  # Regular game play mode
                if agent_obs['ball_owned_player'] == agent_obs['active']:
                    # Calculate shot accuracy reward
                    if np.linalg.norm(agent_obs['ball_direction']) < self.shot_accuracy_threshold:
                        new_rewards[idx] += self.shot_reward_multiplier
                        components["shot_accuracy_reward"][idx] = self.shot_reward_multiplier
                    
                    # Calculate dribble efficiency reward
                    if 'dribble' in agent_obs['sticky_actions'] and np.random.rand() < self.dribble_efficiency_threshold:
                        new_rewards[idx] += self.dribble_reward_multiplier
                        components["dribble_efficiency_reward"][idx] = self.dribble_reward_multiplier
                    
                    # Estimate pass accuracy reward based on pass success in a window of opportunity
                    if 'long_pass' in agent_obs['sticky_actions'] or 'high_pass' in agent_obs['sticky_actions']:
                        if np.random.rand() < self.pass_accuracy_threshold:
                            new_rewards[idx] += self.pass_reward_multiplier
                            components["pass_accuracy_reward"][idx] = self.pass_reward_multiplier

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        total_reward = sum(reward)
        info["final_reward"] = total_reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
