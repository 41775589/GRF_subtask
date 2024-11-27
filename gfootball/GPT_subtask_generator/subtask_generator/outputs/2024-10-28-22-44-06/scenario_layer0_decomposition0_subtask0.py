from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  # Set the initial ball position
  builder.SetBallPosition(0.0, 0.0)

  # Set the team for training attacking actions
  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.2, e_PlayerRole.CF)
  builder.AddPlayer(-1.0, -0.2, e_PlayerRole.RM)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole.CM)

  # Set the opponent team (for realism)
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole.GK)
  builder.AddPlayer(0.5, 0.2, e_PlayerRole.LB, lazy=True)
  builder.AddPlayer(0.5, -0.2, e_PlayerRole.CB, lazy=True)
