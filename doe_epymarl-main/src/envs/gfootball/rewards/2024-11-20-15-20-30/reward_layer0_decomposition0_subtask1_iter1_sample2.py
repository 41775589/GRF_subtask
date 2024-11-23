import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A revised wrapper that improves rewards based on defense and tactical actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Tuning the coefficients to balance the rewards based on analysis
        self.defensive_actions_reward = 0.3  # Lowered reward as it was dominating
        self.positioning_reward_coefficient = 0.1  # Reward for good positioning
        self.ball_intercept_reward = 0.2  # Reward for moving towards the ball

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped._cached_observation
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "defensive_action_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "ball_intercept_reward": [0.0] * len(reward)
        }

        for index, agent_obs in enumerate(observation):
            # Tactical positioning reward logic
            if agent_obs['left_team_active'][index]:
                enemy_goal = np.array([1, 0])  # assuming right side goal is at (1, 0)
                dist_to_goal = np.linalg.norm(agent_obs['left_team'][index] - enemy_goal)
                components["positioning_reward"][index] = max(0, 1 - dist_to_goal) * self.positioning_reward_coefficient

            # Defensive action reward
            if 'ball_owned_team' in agent_obs and agent_obs['ball_owned_team'] in [1, -1]:  # if the ball is not owned or owned by opposite team
                ball_position = agent_obs['ball'][:2]
                agent_position = agent_obs['left_team'][index]
                dist_to_ball = np.linalg.norm(agent_position - ball_position)
                if dist_to_ball < 0.1:  # close to ball
                    components["ball_intercept_reward"][index] = self.ball_intercept_reward

            # Aggregate rewards into the final adjusted reward
            reward[index] = sum([
                components["base_score_reward"][index],
                components["defensive_action_reward"][index],
                components["positioning_reward"][index],
                components["ball_intercept_reward"][index]
            ])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Evaluate the reward using the custom reward logic
        reward, components = self.reward(reward)
        # Compile the final reward and components for feedback
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
