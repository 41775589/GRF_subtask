from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  
  builder.SetBallPosition(0.0, 0.0)

  # Defensive team
  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(0.0, 0.25, e_PlayerRole_RM)
  builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM)
  builder.AddPlayer(0.0, -0.25, e_PlayerRole_CB)

  # Attacking team with a lazy keeper
  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True)
