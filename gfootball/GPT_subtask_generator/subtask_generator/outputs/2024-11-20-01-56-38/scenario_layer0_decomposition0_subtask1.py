from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Starting ball position is in the defensive half to simulate defending scenarios
    builder.SetBallPosition(-0.3, 0.0)

    # Team e_Left is the team that we are training for defensive maneuvers
    builder.SetTeam(Team.e_Left)
    # Goalkeeper
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Two Center Backs practicing clearing the ball, intercepting, and defensive positioning
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CB)
    # One Defensive Midfielder to practice tackles and interceptions
    builder.AddPlayer(-0.3, 0.0, e_PlayerRole_DM)
    
    # Team e_Right as the attacking opponent
    builder.SetTeam(Team.e_Right)
    # Opponent's formation simulates a pressing attack to test our defense
    builder.AddPlayer(-0.3, 0.0, e_PlayerRole_CF)
    builder.AddPlayer(-0.2, 0.15, e_PlayerRole_RM)
    builder.AddPlayer(-0.2, -0.15, e_PlayerRole_LM)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
