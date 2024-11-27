from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False
    
    # Setting initial ball position closer to the offensive side to encourage direct attacks
    builder.SetBallPosition(0.6, 0.0)

    # Setting up the left team (Our Team focusing on attacking exercises)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.7, -0.1, e_PlayerRole_CF)  # Attacking forward, positioned slightly right
    builder.AddPlayer(0.7, 0.1, e_PlayerRole_CF)   # Attacking forward, positioned slightly left

    # Setting up the right team (Opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    # Defensive players to simulate a realistic game-like pressure on the attackers
    builder.AddPlayer(0.5, -0.1, e_PlayerRole_CB)  # Center Back, slightly to the right
    builder.AddPlayer(0.5, 0.1, e_PlayerRole_CB)   # Center Back, slightly to the left
    builder.AddPlayer(0.4, 0.0, e_PlayerRole_CB)   # Additional Center Back in the middle to increase pressure
