from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting initial ball position near the attacking team to facilitate shooting exercises
    builder.SetBallPosition(0.3, 0.0)

    # Setting up the training team (Left team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Adding two offensive players focusing on dribbling and shooting from different positions
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_AM)  # Attacker in a more central position
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_AM)  # Attacker in a wide position to practice angled shots

    # Setting up the opponent team (Right team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Adding a central defender to provide some opposition and realistic pressure scenarios
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CB)
