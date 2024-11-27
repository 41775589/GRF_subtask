from . import *
def build_scenario(builder):
    builder.config().game_duration = 600  # Sufficient time for play development
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Focus on attacking skills

    if builder.EpisodeNumber() % 2 == 0:
        attack_team = Team.e_Left
        defend_team = Team.e_Right
    else:
        attack_team = Team.e_Right
        defend_team = Team.e_Left

    # Set up attacking team with the specific roles focused on shooting and dribbling
    builder.SetTeam(attack_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)  # Goalkeeper
    builder.AddPlayer(0.500000, -0.200000, e_PlayerRole_CF)  # Center Forward positioned for shots
    builder.AddPlayer(0.500000, 0.200000, e_PlayerRole_AM)  # Attacking Midfielder for dribbling and support

    # Set up defending team with less aggressive positioning
    builder.SetTeam(defend_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)  # Goalkeeper
    builder.AddPlayer(-0.500000, -0.100000, e_PlayerRole_CB)  # Center Back
    builder.AddPlayer(-0.500000, 0.100000, e_PlayerRole_CB)  # Center Back

    # Set initial ball position to emphasize starting in a challenging attacking position
    builder.SetBallPosition(0.4, 0.0)
