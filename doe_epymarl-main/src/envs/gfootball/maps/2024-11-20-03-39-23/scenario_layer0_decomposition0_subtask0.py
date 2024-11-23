from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting the initial ball position near the center, slightly towards the left
    builder.SetBallPosition(-0.2, 0.0)

    # Setting up the left team (controlled team with 3 agents)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CM)  # Central midfielder for dribbling and passing
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_CM) # Another midfielder to assist in control
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CF)   # Forward to receive long passes and possibly shoot

    # Setting up the right team (opponent with minimal players)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Only goal keeper to defend goal area
