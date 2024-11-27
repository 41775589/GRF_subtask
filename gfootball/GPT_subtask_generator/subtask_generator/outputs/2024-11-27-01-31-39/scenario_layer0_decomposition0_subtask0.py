from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.0, 0.0)  # Ball starts at the center

    # Set the scenario for the team with controlled players focusing on passing and ball control
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_CM)   # Midfielder with ball control tasks
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CM)  # Midfielder with passing tasks

    # Set the scenario for the opponent team (right side) focusing on basic positions
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.3, 0.1, e_PlayerRole_CB)  # Defender
    builder.AddPlayer(0.3, -0.1, e_PlayerRole_CB)  # Defender

    # This scenario creates a simplified environment for practicing passing under pressure and controlling the ball efficiently
