import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward focusing on offensive skills."""

    def __init__(self, env):
        super().__init__(env)
        # Define checkpoints that could be approached for evaluating offensive capabilities.
        self._num_checkpoints = 5
        self._checkpoint_dist_increment = 0.1
        self._checkpoint_rewards = np.linspace(0.2, 1.0, self._num_checkpoints)

    def reset(self):
        """Reset the environment and checkpoints."""
        super().reset()
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store checkpoints state."""
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore checkpoints state."""
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modifies the reward based on the proximity to checkpoints and offensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "offensive_action_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_position = o['ball'][:2]  # Assume 'ball' is (x,y,z), we take (x,y)
            opponent_goal = np.array([1, 0]) # Assuming goal location is normalized at (1,0)
            dist_to_goal = np.linalg.norm(ball_position - opponent_goal)
            
            # Check for checkpoint crossing
            for chkpt_id in range(self._num_checkpoints):
                chkpt_threshold = (chkpt_id + 1) * self._checkpoint_dist_increment
                if dist_to_goal < chkpt_threshold:
                    if self._collected_checkpoints.get((rew_index, chkpt_id), False):
                        continue
                    reward[rew_index] += self._checkpoint_rewards[chkpt_id]
                    components["checkpoint_reward"][rew_index] += self._checkpoint_rewards[chkpt_id]
                    self._collected_checkpoints[(rew_index, chkpt_id)] = True
            
            # Reward for offensive actions: dribble, pass in the attacking direction
            if o['ball_owned_player'] == o['active']:
                if 'Long Pass' in o['action']:
                    components["offensive_action_reward"][rew_index] += 0.3
                if 'Dribble' in o['action']:
                    components["offensive_action_reward"][rew_index] += 0.2
                if 'Shot' in o['action']:
                    components["offensive_action_reward"][rew_index] += 0.5
                reward[rew_index] += sum(components["offensive_action_reward"][rew_index])

        return reward, components

    def step(self, action):
        """Step environment and modify reward based on offensive performance."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Add details to info for transparency and potential debugging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
