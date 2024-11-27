from . import *
def build_scenario(builder):
    builder.config().game_duration = 800  # Longer duration for offensive practice
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Keep the ball even if intercepted for continued practice
    
    builder.SetBallPosition(0.3, 0.0)  # Starting position of the ball further up the field

    # Setting up the training team
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.2, 0.1, e_PlayerRole_AM)  # Attacking midfielder role for practice dribbling and shooting
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_CF)  # Centre forward for intense offensive maneuvers

    # Opponent settings
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    builder.AddPlayer(-0.2, 0.15, e_PlayerRole_CB)  # Opponent center back
    builder.AddPlayer(-0.2, -0.15, e_PlayerRole_CB)  # Another opponent center back

    # Using simpler opponent placement to focus on agent training to outmaneuver and execute plays
