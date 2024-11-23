from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the initial ball position near the center slightly towards the left team
    builder.SetBallPosition(0.1, 0.0)

    # Setting up the Left Team (Training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM)  # Playmaker in midfield for constructing attacks
    builder.AddPlayer(0.2, 0.2, e_PlayerRole_CF)  # Forward for aggressive maneuvers and shooting
    builder.AddPlayer(0.2, -0.2, e_PlayerRole_DM)  # Defensive midfielder for robust defending

    # Setting up the Right Team (Opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.3, 0.1, e_PlayerRole_CB)  # Central defender to oppose forward attacks
    builder.AddPlayer(-0.3, -0.1, e_PlayerRole_CB)  # Another central defender for robust defense
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CM)  # Midfielder to increase midfield dynamics
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_CM)  # Additional midfielder for contesting possession

    # This initial scenario ensures interaction and learning for shooting (CF), passing and controlling gameplay (CM), and defensive maneuvers and counters (DM)
