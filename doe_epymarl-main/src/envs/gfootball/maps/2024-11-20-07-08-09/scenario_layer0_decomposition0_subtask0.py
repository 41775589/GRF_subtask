from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting the ball position near the center slightly closer to the opponent's goal
    builder.SetBallPosition(0.3, 0.0)

    builder.SetTeam(Team.e_Left)
    # Adding one goalkeeper for compliance and it won't participate in the drill
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)

    # Three attacking players positioned to simulate a forward attack scenario
    # Player 1 - Central Forward, starting with the ball, mastering dribble and shot
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CF)
    # Player 2 - Right Midfielder, to practice sprint, stop-sprint, and passing
    builder.AddPlayer(0.1, 0.1, e_PlayerRole_RM)
    # Player 3 - Left Midfielder, to practice dribble and position to receive a pass and then shoot
    builder.AddPlayer(0.1, -0.1, e_PlayerRole_LM)

    builder.SetTeam(Team.e_Right)
    # Opposing team's goalkeeper to serve as the challenge for the scenario 
    builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)
    # Adding a single defender to simulate defensive pressure on our attacking trio
    builder.AddPlayer(0.5, 0.0, e_PlayerRole_CB)
