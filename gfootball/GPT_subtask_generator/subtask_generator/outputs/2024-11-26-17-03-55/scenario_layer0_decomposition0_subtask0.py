from . import *
def build_scenario(builder):
    builder.config().game_duration = 600  # Adjusted for more time to practice dribbling and shooting
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Allows continual focus on ball control

    builder.SetBallPosition(0.3, 0.0)  # Starting position that requires dribbling to reach the goal

    # Set up the left team (offensive players to be trained)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.1, 0.1, e_PlayerRole_CM)  # Close to the ball for immediate action
    builder.AddPlayer(0.1, -0.1, e_PlayerRole_CF)  # Forward to practice shots on goal

    # Set up the right team (defenders)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Keeper in the goal
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CB)  # Center back to challenge the dribbler
