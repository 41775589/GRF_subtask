from . import *
def build_scenario(builder):
    builder.config().game_duration = 400  # Episode duration
    builder.config().deterministic = False  # Non-deterministic game for variability in training
    builder.config().offsides = False  # No offsides rule to simplify offensive moves
    builder.config().end_episode_on_score = True  # Episode ends on scoring
    builder.config().end_episode_on_out_of_play = True  # Episode ends if the ball goes out of play
    builder.config().end_episode_on_possession_change = False  # Continue the episode even if possession changes

    # Initial position of the ball close to one of the attacking players
    builder.SetBallPosition(0.2, 0.0)

    # Setting Left team (training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Adding a goalie not involved in training directly
    builder.AddPlayer(0.1, 0.05, e_PlayerRole_CF)  # Attacker 1
    builder.AddPlayer(0.1, -0.05, e_PlayerRole_CF)  # Attacker 2

    # Setting Right team (defensive opponents)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalie
    builder.AddPlayer(-0.1, 0.12, e_PlayerRole_CB)  # Opponent defender
    builder.AddPlayer(-0.1, -0.12, e_PlayerRole_CB)  # Opponent defender

    # This scenario intents to focus on attackers mastering dribbling and shooting
    # in presence of mild opposition from two defenders.
