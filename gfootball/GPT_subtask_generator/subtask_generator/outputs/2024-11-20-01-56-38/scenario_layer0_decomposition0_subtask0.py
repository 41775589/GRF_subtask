from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Set up positions to challenge offensive strategies involving dribbling, shooting, and passing
    builder.SetBallPosition(0.5, 0.0)  # Center the ball in the opponents' half near the box to start with possession

    # Set up the left team (offensive maneuvers training)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper at the goal
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)  # Forward player close to the box
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CF)   # Another forward player symmetrically opposite

    # Set up the right team (defensive setup)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CB)   # Center Back in the middle to challenge our CF
    builder.AddPlayer(0.25, 0.25, e_PlayerRole_LB) # Left Back to cover side dribbles
    builder.AddPlayer(0.25, -0.25, e_PlayerRole_RB) # Right Back to cover side dribbles
