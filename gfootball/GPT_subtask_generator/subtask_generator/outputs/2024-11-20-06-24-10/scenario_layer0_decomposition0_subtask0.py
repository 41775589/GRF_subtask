from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000  # Enough time for practice transitions and set plays
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = False
    builder.config().end_episode_on_possession_change = False

    # Set the initial ball position to simulate a midfield starting point for transitions
    builder.SetBallPosition(-0.05, 0.0)

    # Configure the left team with 3 players practicing transitions
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper, not controllable
    builder.AddPlayer(-0.4, 0.1, e_PlayerRole_CM)  # Central Midfielder, controllable, should act as the initiator
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_LM)  # Left Midfielder, controllable, to practice dribbling and receive passes
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_RM)  # Right Midfielder, controllable, to practice dribbling and receive passes

    # Configure a simple right team with only a goalkeeper and one defender for minimal resistance
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper, not controllable
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CB, controllable=True)  # Central Back, not very active, to provide token opposition
