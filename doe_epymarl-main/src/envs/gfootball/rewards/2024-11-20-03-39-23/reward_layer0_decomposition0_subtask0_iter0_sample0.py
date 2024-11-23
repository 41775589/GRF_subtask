import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a customized reward for mastering ball control and passing accuracy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Keeping a count of successful passes and dribbles
        self.passing_count = 0
        self.dribble_count = 0

    def reset(self):
        self.passing_count = 0
        self.dribble_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['passing_count'] = self.passing_count
        to_pickle['dribble_count'] = self.dribble_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passing_count = from_pickle['passing_count']
        self.dribble_count = from_pickle['dribble_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        modified_reward = []

        for agent_idx, agent_reward in enumerate(reward):
            o = observation[agent_idx]
            dribble_bonus = 0.0
            pass_bonus = 0.0

            if 'sticky_actions' in o:
                # Reward for dribbling successfully
                if o['sticky_actions'][1] == 1:  # Assuming index 1 is dribbling
                    dribble_bonus = 0.2
                    self.dribble_count += 1

                # Reward for successful passing (short or long)
                if o['game_mode'] == 1:  # Assuming 1 is a game mode where pass is possible
                    pass_bonus = 0.5
                    self.passing_count += 1

            total_agent_reward = agent_reward + dribble_bonus + pass_bonus
            modified_reward.append(total_agent_reward)

            # Adding reward components for visibility in debugging
            components.setdefault('dribble_bonus', []).append(dribble_bonus)
            components.setdefault('pass_bonus', []).append(pass_bonus)

        return modified_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the reward() method
        reward, components = self.reward(reward)

        # Add final reward to the 'info'
        info["final_reward"] = sum(reward)
        
        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
