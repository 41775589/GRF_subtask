import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive and transitional reward."""

    def __init__(self, env):
        super().__init__(env)
        self._num_pass_checkpoints = 5
        self._pass_checkpoint_reward = 0.05
        self._intercept_bonus = 0.2
        self._dispossess_bonus = 0.15

    def reset(self):
        """ Reset the environment and the collected information. """
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the current wrapper state for serialization. """
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state of the wrapper based on deserialized information. """
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """ Calculate the reward based on game state and defensive actions. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": list(reward),
                      "intercept_bonus": [0.0] * len(reward),
                      "dispossess_bonus": [0.0] * len(reward),
                      "passing_checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Check if intercept occurred
            if o['game_mode'] in [4, 5]:  # Assume modes 4 and 5 are interception related
                components["intercept_bonus"][rew_index] = self._intercept_bonus
                reward[rew_index] += components["intercept_bonus"][rew_index]

            # Check if dispossess happened
            if o['ball_owned_team'] != 0 and o['ball_owned_team'] == self._prev_ball_owned_team and o['ball_owned_player'] != self._prev_ball_owned_player:
                components["dispossess_bonus"][rew_index] = self._dispossess_bonus
                reward[rew_index] += components["dispossess_bonus"][rew_index]

            # Calculate passing based checkpoint reward
            if o['ball_owned_team'] == 0:  # Team 0 is the controlled team
                ball_pos = o['ball']
                goal_pos = [1, 0]  # Assuming location of opponent goal at (1,0)
                distance = np.linalg.norm(np.array(ball_pos[:2]) - np.array(goal_pos))
                
                checkpoint_index = min(int((1 - distance) / 0.2 * self._num_pass_checkpoints), self._num_pass_checkpoints - 1)
                if checkpoint_index > self._collected_checkpoints.get(rew_index, -1):
                    components["passing_checkpoint_reward"][rew_index] = self._pass_checkpoint_reward
                    reward[rew_index] += components["passing_checkpoint_reward"][rew_index]
                    self._collected_checkpoints[rew_index] = checkpoint_index

            # Store previous state for comparison in the next step
            self._prev_ball_owned_team = o['ball_owned_team']
            self._prev_ball_owned_player = o['ball_owned_player']

        return reward, components

    def step(self, action):
        """ Step the environment and adjust the reward using the reward function. """
        observation, reward, done, info = self.env.step(action)
        
        # Adjust reward function based on custom calculations
        reward, components = self.reward(reward)
        
        # Sum components into the info dictionary for insights
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
