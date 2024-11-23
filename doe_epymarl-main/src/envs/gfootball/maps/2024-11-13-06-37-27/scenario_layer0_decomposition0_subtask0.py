from . import *
def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, Role.CB)
  builder.AddPlayer(0.6, 0.0, Role.CF)
  builder.AddPlayer(0.7, 0.2, Role.CM)
  builder.AddPlayer(0.7, -0.2, Role.CM)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, Role.GK)
  builder.AddPlayer(-0.75, 0.0, Role.CB)
  builder.AddPlayer(-0.75, 0.2, Role.CB)
