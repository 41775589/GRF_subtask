from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting up the ball position for the scenario
    builder.SetBallPosition(0.0, 0.0)

    # Adding players for the Left Team (Training Group)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK, controllable=True)
    builder.AddPlayer(0.1, -0.1, Role.e_PlayerRole_CB, controllable=True)

    # Adding players for the Right Team
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK, controllable=False)  # Non-controllable goalkeeper
    builder.AddPlayer(-0.1, 0.1, Role.e_PlayerRole_CF, controllable=True)  # Attacker for testing defensive actions
