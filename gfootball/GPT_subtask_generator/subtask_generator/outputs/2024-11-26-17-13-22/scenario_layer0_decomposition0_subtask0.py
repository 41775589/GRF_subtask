from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000  # Sufficiently long to practice different skills
    builder.config().right_team_difficulty = 0.0  # Disable AI opponents for focused training
    builder.config().left_team_difficulty = 0.0  # Disable AI opponents for focused training
    builder.config().deterministic = False
    builder.config().offsides = False  # Remove offsides rule to not limit forward movements
    builder.config().end_episode_on_score = True  # Restart scenario on scoring
    builder.config().end_episode_on_out_of_play = True  # Restart scenario if out of play
    builder.config().end_episode_on_possession_change = False  # Keep possession even if missed pass

    # Starting ball position
    builder.SetBallPosition(0.5, 0.0)  # Set the ball near the attacking half

    # Configuration for Team Left
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.4, -0.2, e_PlayerRole_CF)  # Position for practice shooting from the left wing
    builder.AddPlayer(0.4, 0.2, e_PlayerRole_AM)  # Position for practice dribbling and long passes from the right wing

    # Configuration for Team Right
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper to practice shooting against
    # Adding defenders to practice against will increase the realism of breaking defensive lines.
    builder.AddPlayer(0.2, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.2, 0.1, e_PlayerRole_CB)
