from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.SetBallPosition(0.0, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, Role.e_PlayerRole_GK, controllable=True)
  builder.AddPlayer(-0.2, 0.0, Role.e_PlayerRole_CB, controllable=True)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, Role.e_PlayerRole_GK, controllable=True)
  builder.AddPlayer(0.2, 0.0, Role.e_PlayerRole_CB, controllable=True)
