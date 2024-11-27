from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.SetBallPosition(0.5, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK)
  builder.AddPlayer(0.6, 0.2, Role.e_PlayerRole_CF)
  builder.AddPlayer(0.6, -0.2, Role.e_PlayerRole_RM)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK)
  builder.AddPlayer(-0.5, 0.15, Role.e_PlayerRole_CB)
  builder.AddPlayer(-0.5, -0.15, Role.e_PlayerRole_RM)
