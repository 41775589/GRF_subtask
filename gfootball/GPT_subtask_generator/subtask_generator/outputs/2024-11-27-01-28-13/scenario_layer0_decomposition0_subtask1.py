from . import *
def build_scenario(builder):
    builder.config().game_duration = 1000  # Adjust duration to allow ample time for learning defensive actions
    builder.config().deterministic = False
    builder.config().offsides = False  # Disable offsides to focus on defensive skills
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Allow continued play after possession change for more interaction

    # Set initial scenario with the ball close to the opponent's goal area - simulating a defensive recovery situation
    builder.SetBallPosition(-0.6, 0)

    # Setting up the left team (our controlled defensive team with 3 players)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.1, e_PlayerRole_CB)  # Center Back 1
    builder.AddPlayer(-0.5, -0.1, e_PlayerRole_CB)  # Center Back 2
    builder.AddPlayer(-0.3, 0.0, e_PlayerRole_CM)  # Defensive Midfielder to practice passing

    # Setting up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's goalkeeper
    # Adding opponent players in attacking positions to simulate pressure
    builder.AddPlayer(-0.4, 0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.4, -0.15, e_PlayerRole_CF)
    builder.AddPlayer(-0.25, 0.0, e_PlayerRole_AM)  # Attacking midfielder

    # The practice focuses on the left team defending against incoming attackers and mastering the response under pressure.
