import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on defensive play and ball distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.intercept_reward = 0.5
        self.pass_reward = 0.3
        self.positioning_reward = 0.2
    
    def reset(self):
        """Reset the environment and reward tracking."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the environment's state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment's state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Reward the agents for intercepting, effective passes, and maintaining good defensive positioning.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward),
                      "intercept_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            # Check intercepting the ball
            if o['game_mode'] in [2, 4]:  # modes related to interceptions like free-kicks, corners
                if o['ball_owned_team'] == 0:  # if ball is owned by the agent's team
                    components['intercept_reward'][i] = self.intercept_reward
                reward[i] += components['intercept_reward'][i]

            # Check effective passing
            if o['game_mode'] == 0:  # normal play
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    components['pass_reward'][i] = self.pass_reward
                reward[i] += components['pass_reward'][i]

            # Check maintaining positions - simplistic version
            if np.linalg.norm(o['right_team'][o['active']] - o['ball']) < 0.1:  # if agent close to the ball
                components['positioning_reward'][i] = self.positioning_reward
                reward[i] += components['positioning_reward'][i]

        return reward, components

    def step(self, action):
        """Step through environment, get observation, modify reward, and return step."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
