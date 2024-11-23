from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    builder.SetBallPosition(0.0, 0.0)  # Start with the ball in the center

    # Set the left team with two players focusing on ball control and play-making
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CM)  # Attacking midfielder with good ball handling
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_LM)  # Left midfielder to practice passes

    # Set the right team as opposing team with fewer players
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)  # Center back to put pressure

    # Set a scenario that focuses on the development of dribbling, passing in pressure situations
