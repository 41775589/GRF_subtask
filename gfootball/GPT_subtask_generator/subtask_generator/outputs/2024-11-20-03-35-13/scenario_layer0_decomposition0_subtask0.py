from . import *
def build_scenario(builder):
    # Set general configuration for the training scenario
    builder.config().game_duration = 600  # Duration to allow enough time for practice
    builder.config().deterministic = False  # Add variability to each episode
    builder.config().offsides = False  # Simplify the rules to focus on offensive skills
    builder.config().end_episode_on_score = True  # End episode when a goal is scored to reinforce learning
    builder.config().end_episode_on_out_of_play = True  # Ends episode if ball goes out to reset quickly
    builder.config().end_episode_on_possession_change = False  # Continues even if possession is lost to allow more continuous play

    # Set ball position near the mid-field to encourage offensive plays
    builder.SetBallPosition(0.1, 0.0)

    # Left team is the training team
    builder.SetTeam(Team.e_Left)
    # Adding two offensive players: one center forward and one attacking midfielder
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CF)  # Center-Forward positioned centrally, likely to have the initial ball
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_AM)  # Attacking Midfielder positioned slightly back and to the side

    # Right team acts as the opponent
    builder.SetTeam(Team.e_Right)
    # Add opponent players that challenge the trainees in their tasks
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)  # Opponent Goalkeeper
    builder.AddPlayer(-0.7, 0.1, e_PlayerRole_CB)  # Opponent Centre Back to apply defensive pressure
    builder.AddPlayer(-0.6, 0.0, e_PlayerRole_LB)  # Opponent Left Back to provide width in defense
