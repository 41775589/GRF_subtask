from . import *
def build_scenario(builder):
    # General settings for the training scenario
    builder.config().game_duration = 800  # longer duration for comprehensive practice
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # allows continued practice even if possession is lost

    builder.SetBallPosition(0.0, 0.0)  # start from center

    # Set up the Left Team (our team with 2 players learning technical skills)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper must be added but is not controllable
    builder.AddPlayer(0.1, 0.0, e_PlayerRole_AM)  # Attacking midfielder to practice dribbling and shooting
    builder.AddPlayer(0.0, 0.2, e_PlayerRole_CF)  # Centre forward to practice passes and finishing

    # Set up the Right Team (opponent with basic setup)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper, non-controllable
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CB)  # Standard defender to act as an obstacle
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_DM)  # Defensive midfielder to add pressure
