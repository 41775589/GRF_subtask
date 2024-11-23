from . import *
def build_scenario(builder):
    builder.config().game_duration = 1200  # Adjust the game duration to provide ample time for pass practice
    builder.config().deterministic = False
    builder.config().offsides = True  # Enable offsides to make the scenario more realistic
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Do not end on possession change to allow continuous play

    builder.SetBallPosition(0.0, 0.0)  # Start with the ball at center field

    # Set up the left (controlled) team
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_CM)  # Center Midfielder with ball
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_CM)  # Center Midfielder

    # Set up the right (opponent) team
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CB, lazy=True)  # Opponent Center Back to offer passive resistance
