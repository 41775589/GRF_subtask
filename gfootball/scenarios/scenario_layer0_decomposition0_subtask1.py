from . import *
def build_scenario(builder):
    # Set general configurations for the scenario
    builder.config().game_duration = 800
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False
    builder.config().deterministic = False

    # Set ball position in the middle of the field
    builder.SetBallPosition(0.0, 0.0)

    # Configure the left team (defensive team with our 3 agents)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CB)  # Center Back right
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CB) # Center Back left
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_DM)  # Defensive Midfielder

    # Configure the right team (opponent, to challenge defenses)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper (non-controllable)
    # Add strikers to simulate attack on our defense
    builder.AddPlayer(0.3, 0.2, e_PlayerRole_CF, controllable=False)  # Right Striker
    builder.AddPlayer(0.3, -0.2, e_PlayerRole_CF, controllable=False) # Left Striker
    builder.AddPlayer(0.15, 0.0, e_PlayerRole_AM, controllable=False) # Attacking Midfielder
