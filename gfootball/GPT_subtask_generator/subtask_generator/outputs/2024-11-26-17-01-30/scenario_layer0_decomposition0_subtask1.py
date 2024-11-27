from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetBallPosition(0.0, 0.0)

  # Group 1: Defensive Actions
  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.07, Role.CB)  # Center Back for marking
  builder.AddPlayer(-1.0, -0.07, Role.LB)  # Left Back for blocking passes and interceptions

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, Role.GK, lazy=True)  # Opponent Goalkeeper to test the defense

  # Fake players for empty team in case there are no players
  builder._FakePlayersForEmptyTeam(builder._scenario_cfg.left_team)
  builder._FakePlayersForEmptyTeam(builder._scenario_cfg.right_team)
