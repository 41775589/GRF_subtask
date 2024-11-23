from . import *
def build_scenario(builder):
    builder.config().game_duration = 800  # Extended duration for more practice
    builder.config().deterministic = False
    builder.config().offsides = False  # Allow for a more open game format
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Keeping possession even if the ball is lost temporarily

    builder.SetBallPosition(0.5, 0.0)  # Moderate field position to practice long passes and shots

    # Setting up the Left Team (Active training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.3, 0.1, e_PlayerRole_CF, controllable=True)  # Center Forward, active role in shooting
    builder.AddPlayer(0.3, -0.1, e_PlayerRole_AM, controllable=True)  # Attacking Midfielder, for dribbling and long passes

    # Setting up the Right Team (Opposition)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0, 0, e_PlayerRole_CB)  # Central defender to oppose the attacks
    builder.AddPlayer(-0.3, 0.1, e_PlayerRole_CM)  # Central Midfielder, to apply pressure
    builder.AddPlayer(-0.3, -0.1, e_PlayerRole_CM)  # Another Central Midfielder
