import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adjusts the reward function to incentivize attacking actions 
    like shooting, dribbling towards the goal, and sprinting past defenders.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define coefficients for additional rewards
        self._goal_shot_reward = 2.0
        self._dribble_toward_goal_reward = 1.0
        self._sprint_bonus = 0.5

    def reset(self):
        """ Resets the environment and necessary components. """
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Retrieves state information with additional data added by the wrapper. """
        to_pickle['CheckpointRewardWrapper'] = {'goal_shot_reward': self._goal_shot_reward}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Sets the state from external source, with consideration of wrapper's additions. """
        from_pickle = self.env.set_state(state)
        self._goal_shot_reward = from_pickle['CheckpointRewardWrapper']['goal_shot_reward']
        return from_pickle

    def reward(self, reward):
        """
        Augments the reward based on attacking gameplay.
        Takes the original reward list and modifies it based on the observed actions.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goal_shot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]  # observation for current agent
            ball_pos = o['ball']
            # proximity to goal (x coordinate closer to 1 means closer to opponent's goal)
            proximity_to_goal = ball_pos[0]

            if o['sticky_actions'][7]:  # Sprint
                reward[rew_index] += self._sprint_bonus
                components['sprint_reward'][rew_index] = self._sprint_bonus

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # player has the ball, check for shot at goal or moving towards goal
                if abs(ball_pos[0] - 1) < 0.1:  # near opponent goal
                    reward[rew_index] += self._goal_shot_reward * proximity_to_goal
                    components['goal_shot_reward'][rew_index] = self._goal_shot_reward * proximity_to_goal

                # rewards for moving towards the goal with the ball
                if ball_pos[0] > 0:  # player is on the opponent's side
                    reward[rew_index] += self._dribble_toward_goal_reward * proximity_to_goal
                    components['dribble_reward'][rew_index] = self._dribble_toward_goal_reward * proximity_to_goal

        return reward, components

    def step(self, action):
        """ Step function applying the new reward mechanism by calling the reward method. """
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the custom reward function
        reward, components = self.reward(reward)
        # Sum up the reward for info
        info["final_reward"] = sum(reward)
        # Add details of components to info for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
