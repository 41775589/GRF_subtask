import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper designed to enhance the training of the subtask focusing on ball control,
    dribbling, sprinting, and precision in passing actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # We will monitor the distance the ball travels while under control for dribbling
        # And keep track of pass completions under pressure, where the agent must execute accurate Short and High Passes.
        self.pass_completion_count = 0
        self.dribble_control_strength = 0

    def reset(self):
        self.pass_completion_count = 0
        self.dribble_control_strength = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['pass_completion_count'] = self.pass_completion_count
        to_pickle['dribble_control_strength'] = self.dribble_control_strength
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_completion_count = from_pickle.get('pass_completion_count', 0)
        self.dribble_control_strength = from_pickle.get('dribble_control_strength', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": np.zeros(len(reward)),
            "pass_completion_reward": np.zeros(len(reward))
        }
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            
            # Reward dribbling when ball is controlled and player is moving.
            if o['ball_owned_team'] == 0 and o['active']:
                self.dribble_control_strength += 0.1  # Increase dribble control
                components["dribble_reward"][i] = self.dribble_control_strength
            
            # Reward successful passes under pressure
            if 'high_pass' in o['sticky_actions'] or 'short_pass' in o['sticky_actions']:
                self.pass_completion_count += 1
                components["pass_completion_reward"][i] = 0.1 * self.pass_completion_count
                
            # Combine rewards into a single reward array
            reward[i] += components["dribble_reward"][i] + components["pass_completion_reward"][i]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward method
        reward, components = self.reward(reward)
        # Add final reward to the info for diagnostics
        info["final_reward"] = sum(reward)
        # Add detailed reward components to info
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
