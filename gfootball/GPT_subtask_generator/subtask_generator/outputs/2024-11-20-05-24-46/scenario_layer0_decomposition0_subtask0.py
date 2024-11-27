from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.SetBallPosition(0.5, 0.0)  # Positions the ball near the opponent's half to encourage attacking plays

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)  # Central Forward (Striker)
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_AM)  # Attacking Midfielder
    
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    # Defenders to provide pressure and simulate defensive scenarios
    builder.AddPlayer(0.5, -0.15, e_PlayerRole_CB) # Center Back
    builder.AddPlayer(0.5, 0.15, e_PlayerRole_CB)  # Center Back

    # Set a scenario where attackers have opportunities both to dribble towards goal under pressure
    # and to execute strategic passes (short, long, high) to find the best approach to score.
