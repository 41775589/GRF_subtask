from . import *
def build_scenario(builder):
    builder.config().game_duration = 600  # Extended duration for practicing various offensive tasks
    builder.config().deterministic = False
    builder.config().offsides = False  # Disable offsides to focus on dribbling and shooting skills
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False  # Continue after possession changes to increase engagement

    builder.SetBallPosition(0.5, 0.0)  # Ball starts near the midfield to facilitate quick offensive plays

    # Left Team: Our controlled players
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)  # Attacking forward with dribbling tasks
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CF)  # Another attacking forward focused on shooting

    # Right Team: Opponents
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent goalkeeper
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_CB)  # Central back to defend against our forwards
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CB)  # Another central back to increase the defensive challenge

    # This setting allows players to continually practice attacking maneuvers against a structured defense
    # without continuous stoppage, thus maximizing the learning on offensive tactics stated in the subtask.
