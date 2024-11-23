import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a positional reward targeting the creation and exploitation of scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.goal_position = np.array([1, 0])  # Imaginary position of the goal in normalised coordinates on the field

    def reset(self):
        """Reset the environment and any accumulated state."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add any state elements that should be pickled to save the environment's state."""
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        """Set the environment's state from unpickled state."""
        self.env.set_state(state)

    def reward(self, reward):
        """Modify reward to emphasize tactical advancement and opportunity creation."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward), "position_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'left_team' in observation[rew_index]:  # Using the left team positions
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    # Player active and has the ball
                    player_position = o['left_team'][o['active']]
                    goal_direction = self.goal_position - player_position
                    goal_distance = np.linalg.norm(goal_direction)
                    components['position_reward'][rew_index] = max(0, 1 - goal_distance) * 0.1
                    reward[rew_index] += components['position_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Take an action and modify the reward then return the results."""
        obs, reward, done, info = self.env.step(action)
        modified_reward, reward_components = self.reward(reward)
        info['final_reward'] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        return obs, modified_reward, done, info
