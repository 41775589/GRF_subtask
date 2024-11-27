from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False
    
    builder.SetBallPosition(0.0, 0.0)

    # Set up the left team (training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.3, 0.1, e_PlayerRole_CM)  # Central midfielder, focusing on passes
    builder.AddPlayer(-0.3, -0.1, e_PlayerRole_LM)  # Left midfielder, focusing on passes and movement
    builder.AddPlayer(-0.3, 0.0, e_PlayerRole_RM)  # Right midfielder, focusing on passes and movement

    # Set up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.2, 0.2, e_PlayerRole_CB, lazy=True)  # Right-center back, defensive, non-active
    builder.AddPlayer(0.2, -0.2, e_PlayerRole_CB, lazy=True)  # Left-center back, defensive, non-active

    # This setup aims for the training team to frequently attempt passing exercises while 
    # defending against an occasionally passive opponent, which moves minimally or not at all.
