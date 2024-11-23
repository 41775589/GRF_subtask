from . import *
def build_scenario(builder):
    builder.config().game_duration = 3000
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.3, 0.0)  # Midfield position slightly towards the opponent's half

    # Setting up the left team which is under training
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.1, 0.0, e_PlayerRole_CM)  # Central Midfielder with the ball
    builder.AddPlayer(0.5, 0.1, e_PlayerRole_RM)  # Right Midfielder, to simulate short and long passes
    builder.AddPlayer(0.5, -0.1, e_PlayerRole_LM)  # Left Midfielder, to simulate short and long passes

    # Setting up the right team as simple passive obstacles
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.6, 0.1, e_PlayerRole_CB, lazy=True)  # Passive Center Back
    builder.AddPlayer(0.6, -0.1, e_PlayerRole_CB, lazy=True)  # Passive Center Back
    
    # Focusing on dribbling, passing and shooting for the left team players
