from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = True
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    # Positioning the ball near the defensive team to simulate pressing
    # and transitioning from defense to attack
    builder.SetBallPosition(-0.6, 0)

    # Setting up the left team with controlled defensive players
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1, 0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.6, -0.1, e_PlayerRole_CB)  # Centre Back left
    builder.AddPlayer(-0.6, 0.1, e_PlayerRole_CB)  # Centre Back right

    # Opponents on the right team
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1, 0, e_PlayerRole_GK)  # Goalkeeper
    # Attacker from the opposing team positioned to challenge the defenders
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CF)  # Forward center
    builder.AddPlayer(-0.5, -0.3, e_PlayerRole_CM)  # Midfielder left
    builder.AddPlayer(-0.5, 0.3, e_PlayerRole_CM)  # Midfielder right
