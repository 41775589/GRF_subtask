from . import *
def build_scenario(builder):
    # Set up basic configuration.
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Initial ball position set near the center to facilitate practice of passing and dribbling.
    builder.SetBallPosition(0.0, 0.0)

    # Setting up the left team with two players for mastering ball control skills.
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_GK, controllable=False)  # Non-controllable goalkeeper.
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_CF) # Controllable forward for dribbling and shooting practice.
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CM) # Controllable midfielder for passing drills.

    # The right team with defensive set up to facilitate real-game scenarios for the left team.
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Non-controllable goalkeeper.
    builder.AddPlayer(-0.7, 0.0, e_PlayerRole_CB, controllable=False)  # Non-controllable defender to act as an obstacle.
    builder.AddPlayer(-0.7, 0.1, e_PlayerRole_CB, controllable=False)  # Non-controllable defender to act as an obstacle.
