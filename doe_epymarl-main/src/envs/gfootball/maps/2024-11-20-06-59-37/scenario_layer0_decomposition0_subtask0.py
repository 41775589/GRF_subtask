from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Allow extended play to practice passing and dribbling

    builder.SetBallPosition(0.0, 0.0)  # Start with the ball in the center for a neutral setup
    
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CF)  # Center forward to practice shooting
    builder.AddPlayer(0.1, 0.2, e_PlayerRole_RM)  # Right midfielder to practice dribbling and passing
    builder.AddPlayer(0.1, -0.2, e_PlayerRole_LM)  # Left midfielder to practice dribbling and passing
    
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opposing goalkeeper
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CB)  # Opposing center back to provide defensive pressure
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_CB)  # Another opposing center back to increase defensive challenge
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_DM)  # Defensive midfielder to challenge midfield plays
