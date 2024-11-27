class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that promotes skilful ball control for offensive plays."""
    def __init__(self, env):
        super().__init__(env)
        # Initialize starting positions for checkpoints distributed in the offensive half.
        self.checkpoints = [
            (0.5, 0.0), (0.75, 0.25), (0.75, -0.25), (0.9, 0.1), 
            (0.9, -0.1), (1.0, 0.0)  # Near and around the goal area
        ]
        self.checkpoint_values = np.ones(len(self.checkpoints)) * 0.1  # Reward value for each checkpoint
        self.visited_checkpoints = {i: False for i in range(len(self.checkpoints))}

    def reset(self, **kwargs):
        self.visited_checkpoints = {i: False for i in range(len(self.checkpoints))}
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Ensure the observation is structured as expected
        if observation is None or 'ball' not in observation:
            return reward

        ball_pos = observation['ball']
        ball_owned_team = observation['ball_owned_team']

        # Reward only if our team controls the ball
        if ball_owned_team == 0:
            player_pos = observation['left_team'][observation['active']]
            for i, checkpoint in enumerate(self.checkpoints):
                if not self.visited_checkpoints[i] and self._is_close_to(checkpoint, player_pos, threshold=0.1):
                    reward += self.checkpoint_values[i]
                    self.visited_checkpoints[i] = True

        return reward

    def _is_close_to(self, target_pos, player_pos, threshold=0.05):
        """Check if the player is close enough to a target checkpoint."""
        distance = np.sqrt((target_pos[0] - player_pos[0])**2 + (target_pos[1] - player_pos[1])**2)
        return distance <= threshold
