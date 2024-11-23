from . import *
def build_scenario(builder):
    # Configure the overall game settings
    builder.config().game_duration = 800
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    # Set initial ball position for offensive scenario starting near midfield
    builder.SetBallPosition(0.3, 0.0)
    
    # Configure the left team, where our trained agents are
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.2, 0.1, e_PlayerRole_CF)   # Center forward, main agent
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_AM)  # Attacking midfielder, secondary agent
    
    # Configure the right team, simulate basic defensive opponents
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB)   # Center back
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CB)  # Another center back
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_DM)   # Defensive midfielder
