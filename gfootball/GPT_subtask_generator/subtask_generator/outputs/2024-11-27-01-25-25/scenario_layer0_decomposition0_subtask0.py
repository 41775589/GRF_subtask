from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False

    builder.SetBallPosition(0.0, 0.0)

    # Setting the team to e_Left and adding three players with roles focusing on ball control and passing
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CM)  # Central Midfielder with ball control
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CM)  # Central Midfielder for passing
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CM) # Central Midfielder for dribbling

    # Setting up an opposing team with less aggressive configurations to allow developing possession skills
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_CB)   # Opponent Center Back, not very aggressive
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CB)  # Opponent Center Back, not very aggressive
