import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for ball control skills, focusing on dribbling, passing, and shooting."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Reward multipliers
        self.position_reward_coefficient = 1.0
        self.shot_reward_coefficient = 3.0
        self.pass_reward_coefficient = 2.0
        self.dribble_reward_coefficient = 1.5

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "position_reward": [0.0, 0.0],
                      "shot_reward": [0.0, 0.0],
                      "pass_reward": [0.0, 0.0],
                      "dribble_reward": [0.0, 0.0]}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for agent_idx in range(len(reward)):
            obs = observation[agent_idx]

            if obs['ball_owned_team'] == 0:  # Team 0
                player_idx = obs['ball_owned_player']
                if player_idx == obs['active']:  # Active player has the ball
                    # Position based reward
                    goal_position = np.array([1, 0, 0])  # Example goal position
                    ball_position = np.array(obs['ball'])
                    distance_to_goal = np.linalg.norm(ball_position[:2] - goal_position[:2])
                    position_reward = self.position_reward_coefficient / (distance_to_goal + 1e-4)
                    components['position_reward'][agent_idx] += position_reward

                    # Action based rewards
                    actions = obs['sticky_actions']
                    if actions[1] == 1:  # Assuming index 1 is shooting
                        components['shot_reward'][agent_idx] = self.shot_reward_coefficient
                    if actions[2] == 1:  # Assuming index 2 is passing
                        components['pass_reward'][agent_idx] = self.pass_reward_coefficient
                    if actions[3] == 1:  # Assuming index 3 is dribbling
                        components['dribble_reward'][agent_idx] = self.dribble_reward_coefficient

            # Aggregate rewards
            aggregated_reward = sum(components[key][agent_idx] for key in components)
            reward[agent_idx] += aggregated_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
