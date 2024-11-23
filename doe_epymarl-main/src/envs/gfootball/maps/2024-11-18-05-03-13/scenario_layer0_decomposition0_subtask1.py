from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)

    # Position the players to simulate tactical movement and teamwork with passing.
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Two midfielders to focus on passing and positional play
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CM)

    builder.SetTeam(second_team)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    # Defensive setup to force our training agents to pass and find space.
    builder.AddPlayer(-0.4, 0.2, e_PlayerRole_CB)  
    builder.AddPlayer(-0.4, -0.2, e_PlayerRole_CB)  
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_DM)  # Defensive midfielder to apply pressure
