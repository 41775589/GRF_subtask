import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for tactical positioning and defensive actions."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._defensive_zones = 10
        self._zone_reward = 0.05

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "zone_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            if o['game_mode'] == 0:  # Only reward during normal play
                # Calculate distance to zones
                if self._defensive_zones:
                    player_pos = o['left_team'][o['active']] if 'left_team' in o else o['right_team'][o['active']]
                    ball_pos = o['ball'][:2]  # Ignore z-coordinate

                    distance_to_ball = np.linalg.norm(player_pos - ball_pos)

                    # Incentivize players to stay close to the ball as a defensive strategy
                    zone_threshold = 0.1 * self._defensive_zones
                    if distance_to_ball < zone_threshold:
                        components["zone_reward"][i] = self._zone_reward
                        reward[i] += components["zone_reward"][i]
            
            # Add efforts to tackle or contest the ball possession when near the opponent
            if o['sticky_actions'][8] == 1:  # Checking if sprint action is active
                components["zone_reward"][i] += 0.02
                reward[i] += 0.02

            if o['sticky_actions'][6] == 1 or o['sticky_actions'][7] == 1:  # Sliding or dribble tackling
                components["zone_reward"][i] += 0.03
                reward[i] += 0.03
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
