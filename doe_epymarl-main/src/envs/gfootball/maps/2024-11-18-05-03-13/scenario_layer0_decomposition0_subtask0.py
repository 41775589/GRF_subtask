from . import *
def build_scenario(builder):
    builder.config().game_duration = 600
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    # Start with ball in midfield and one agent handling the ball
    builder.SetBallPosition(0, 0)

    # Setting up the left team (control group with 3 players)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CF)   # Center Forward with the ball
    builder.AddPlayer(0.2, 0.1, e_PlayerRole_CM)   # Center Midfielder
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_CM)  # Center Midfielder

    # Right team as simple opponents
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.15, e_PlayerRole_CB)  # Central Back near the corner of the penalty area
    builder.AddPlayer(-0.5, -0.15, e_PlayerRole_CB)  # Central Back near the other corner of the penalty area

    # Another Central Midfielder positioned further as an obstacle if playing from midfield
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CM)
