from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Initial positions of three attacking players in a more aggressive setup
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.7, 0.0, e_PlayerRole_CF)  # Central Forward
    builder.AddPlayer(0.7, 0.2, e_PlayerRole_LM)  # Left Midfielder
    builder.AddPlayer(0.7, -0.2, e_PlayerRole_RM)  # Right Midfielder

    # Opponent team configuration with less aggressive, more defensive setup
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CB)  # Center Back
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)  # Center Back
    builder.AddPlayer(-0.5, 0.3, e_PlayerRole_LB)  # Left Back
    builder.AddPlayer(-0.5, -0.3, e_PlayerRole_RB)  # Right Back

    # Set the ball position to foster attacking plays
    builder.SetBallPosition(0.5, 0.0)
