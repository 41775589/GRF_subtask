from . import *
def build_scenario(builder):
    # Set the basic configuration for the training scenario
    builder.config().game_duration = 600  # Slightly longer to allow for dribbling and shooting practice
    builder.config().deterministic = False
    builder.config().offsides = False  # Disable offsides to focus on dribbling and shooting skills
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Maintain possession to practice dribbling

    # Set ball position near the middle, slightly towards the opponent's goal
    builder.SetBallPosition(0.2, 0.0)

    # Configure the team on the left (team that will be performing the training task)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Add a goalkeeper
    builder.AddPlayer(0.1, -0.1, e_PlayerRole_CM)  # Midfielder starting with the ball
    builder.AddPlayer(0.1, 0.1, e_PlayerRole_CF)  # Forward to practice close-range shots

    # Configure the opposing team on the right
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    builder.AddPlayer(0.0, -0.15, e_PlayerRole_CB)  # Opponent center-back to challenge dribbles
    builder.AddPlayer(0.0, 0.15, e_PlayerRole_CB)  # Another opponent center-back

    # This scenario setup encourages the training of dribbling, maintaining possession under pressure,
    # and executing precise close-range shots. The presence of two defenders will challenge the players
    # to use dribbling (Dribble) and speed (Sprint) effectively to create scoring opportunities and
    # practice stopping the dribble to change attack style or shoot at goal.
