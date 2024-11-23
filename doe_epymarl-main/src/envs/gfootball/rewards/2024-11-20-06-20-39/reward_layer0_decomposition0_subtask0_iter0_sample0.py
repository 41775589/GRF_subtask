import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on ball control, offensive plays, and effective ball progression."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.progress_checkpoints = np.linspace(-1, 1, num=10)  # Represents checkpoints from one end of the field to the other
        self.checkpoint_values = np.zeros((3, 10))  # Reset these values for each episode for each agent

    def reset(self):
        """Reset the checkpoint values for a new episode."""
        self.checkpoint_values = np.zeros((3, 10))
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_values
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.checkpoint_values = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the rewards based on ball control, distance progression, and successful offensive plays."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'control_progression_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            ball_x = o['ball'][0]  # Ball's x-coordinate

            # Update checkpoints based on ball position controlled by the active player
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # if the current agent owns the ball
                # Determine the nearest checkpoint not yet reached
                indices = np.where(self.progress_checkpoints >= ball_x)[0]
                if len(indices) > 0:
                    nearest_checkpoint = indices[0]
                    if not self.checkpoint_values[i, nearest_checkpoint]:  # if checkpoint not yet reached
                        self.checkpoint_values[i, nearest_checkpoint] = 1  # Mark it as reached
                        components['control_progression_reward'][i] += 0.1  # Reward for reaching a new checkpoint

            # Additional rewards for passing, shooting based on the game scenario
            if 'sticky_actions' in o:
                if o['sticky_actions'][8] == 1:  # If doing a high pass
                    components['control_progression_reward'][i] += 0.05
                if o['sticky_actions'][9] == 1 or o['sticky_actions'][10] == 1:  # If doing a shot
                    components['control_progression_reward'][i] += 0.2

            reward[i] += components['control_progression_reward'][i]

        return reward, components

    def step(self, action):
        """Process environment step with modified rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add final reward and component values to info
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
