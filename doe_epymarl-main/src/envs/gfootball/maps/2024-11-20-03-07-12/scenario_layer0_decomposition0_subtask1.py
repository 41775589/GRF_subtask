from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000  # Increased duration for extended play to practice defensive skills
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Keeps the game going even if possession changes to practice regaining control

    builder.SetBallPosition(0.0, 0.0)  # Central position to start practice of interceptions and passing

    # Setting up the Left team with 3 agents focusing on defensive tasks
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.50, 0.10, e_PlayerRole_DM, controllable=True)  # Defensive Midfielder as a key defensive role
    builder.AddPlayer(-0.50, -0.10, e_PlayerRole_CM, controllable=True)  # Center Midfielder to control the midfield

    # Setting up a simpler Right team to act as opponents, not focusing on subtasks
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.50, 0.20, e_PlayerRole_CM, controllable=False)  # Automate some opponent players for interaction
    builder.AddPlayer(0.50, -0.20, e_PlayerRole_CM, controllable=False)
