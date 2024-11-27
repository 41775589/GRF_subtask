from . import *
def build_scenario(builder):
    builder.config().game_duration = 600  # Longer duration for intensive offensive plays
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Allows continuous play even if possession is lost
    
    builder.SetBallPosition(0.0, 0.0)  # Place the ball in midfield to start the scenario

    # Set Team Left as the offensive team with two attackers
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.2, 0.05, e_PlayerRole_CF)  # Centre Forward, primary attacker
    builder.AddPlayer(0.2, -0.05, e_PlayerRole_AM)  # Attacking Midfielder, supporting attack

    # Set Team Right with basic defensive setup, to enable offensive plays
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CB)  # Centre Back to challenge attackers
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CB)  # Another Centre Back for defense
