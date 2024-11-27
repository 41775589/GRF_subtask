import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function specifically for training the 
    attacking skills with action focus on 'Shot', 'Dribble', 'Sprint', and 'Stop-Dribble'.
    Emphasis is placed on reaching closer to the opponent's goal with the ball, 
    and successfully executing shot actions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define rewards for specific actions and situations
        self.shot_reward = 1.0          # Reward for taking a shot
        self.goal_reward = 5.0          # Additional reward for scoring a goal
        self.dribble_reward = 0.2       # Reward for dribbling closer to the goal
        self.sprint_reward = 0.1        # Reward for sprinting with the ball
        self.stop_dribble_reward = 0.1  # Reward for controlling the play near the opponent's goal
        
        # Control how close player needs to get to the goal to obtain positional rewards
        self.goal_distance_threshold = 0.2

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
        
        if observation is None:
            return reward, {}

        new_rewards = []
        reward_components = {"base_score_reward": reward.copy()}

        for obs in observation:
            # Extract necessary details from the observation
            ball_position = obs['ball'][:2]  # get ball x, y position
            goal_position = [1, 0]  # Assuming goal is at x=1, y=0
            ball_owned_team = obs['ball_owned_team']
            ball_owned_player = obs['ball_owned_player']
            active_player = obs['active']
            game_mode = obs['game_mode']

            # Calculate distance to the goal
            distance_to_goal = np.linalg.norm(np.array(ball_position) - np.array(goal_position))

            # Initialize component rewards
            shot_reward = 0
            dribble_reward = 0
            sprint_reward = 0
            stop_dribble_reward = 0
            goal_score_reward = 0

            # Check if the ball is owned by the active player of the team
            if ball_owned_team == 1 and ball_owned_player == active_player:
                if game_mode == 0:  # Normal play mode
                    if 'Shot' in obs['sticky_actions']:
                        shot_reward += self.shot_reward
                        if distance_to_goal < self.goal_distance_threshold:
                            # Assume that shooting near the goal increases chance of scoring
                            goal_score_reward += self.goal_reward
                    if 'Dribble' in obs['sticky_actions']:
                        dribble_reward += self.dribble_reward
                    if 'Sprint' in obs['sticky_actions']:
                        sprint_reward += self.sprint_reward
                    if 'Stop-Dribble' in obs['sticky_actions']:
                        stop_dribble_reward += self.stop_dribble_reward

            # Calculate new reward
            new_reward = reward + shot_reward + dribble_reward + sprint_reward + stop_dribble_reward + goal_score_reward

            # Collect reward components
            reward_components['shot_reward'] = shot_reward
            reward_components['dribble_reward'] = dribble_reward
            reward_components['sprint_reward'] = sprint_reward
            reward_components['stop_dribble_reward'] = stop_dribble_reward
            reward_components['goal_score_reward'] = goal_score_reward

            new_rewards.append(new_reward)

        return new_rewards, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
