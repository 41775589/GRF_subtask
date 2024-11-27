from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  builder.SetBallPosition(0.02, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK)
  builder.AddPlayer(0.5, 0.1, Role.e_PlayerRole_CF)
  builder.AddPlayer(0.5, 0.0, Role.e_PlayerRole_CM)
  builder.AddPlayer(0.5, -0.1, Role.e_PlayerRole_CM)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK)
  builder.AddPlayer(-0.75, 0.0, Role.e_PlayerRole_CB)
  builder.AddPlayer(-0.7, 0.1, Role.e_PlayerRole_RM)
  builder.AddPlayer(-0.7, -0.1, Role.e_PlayerRole_CB)
