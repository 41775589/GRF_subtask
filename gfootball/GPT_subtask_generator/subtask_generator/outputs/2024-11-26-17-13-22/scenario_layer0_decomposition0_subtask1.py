from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting the ball position to a neutral area to allow for building up play.
    builder.SetBallPosition(0.2, 0.0)

    # Setting up the left team with defensive and midfield roles.
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_DM)  # Defensive Midfielder
    builder.AddPlayer(-0.5,  0.1, e_PlayerRole_DM)  # Defensive Midfielder
    builder.AddPlayer(-0.3,  0.0, e_PlayerRole_CM)  # Centre Midfielder

    # Setting up the right team with attacking roles to challenge the midfield and defense.
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.3,  0.1, e_PlayerRole_CF)  # Centre Forward
    builder.AddPlayer(0.3, -0.1, e_PlayerRole_AM)  # Attacking Midfielder
    builder.AddPlayer(0.5,  0.0, e_PlayerRole_CF)  # Centre Forward
