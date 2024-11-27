from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial ball position
    builder.SetBallPosition(0.5, 0.0)

    # Set up the left team (controlled players focused on offense)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)  # Center Forward positioned slightly left
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CF)   # Center Forward positioned slightly right

    # Set up the right team (opponents)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Adding defenders to challenge the offensive skills
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)  # Center Back positioned slightly left
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CB)   # Center Back positioned slightly right
