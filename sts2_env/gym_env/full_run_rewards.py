"""Bounded, undiscounted progress rewards shared by both full-run policies."""
REWARD_VERSION = 2
REWARD_CONFIG = {'version': REWARD_VERSION, 'gamma': 1.0, 'terminal_win': 1.0,
                 'terminal_loss': -1.0, 'floor_bonus_cap': .25,
                 'floor_normalization': 60, 'round_cost': .001, 'round_cost_cap': .10}


class FullRunRewards:
    def __init__(self, initial_floor=0):
        self.highest_floor = initial_floor
        self.floor_credit = 0.0
        self.round_cost = 0.0
        self.finished = False

    def advance(self, floor, rounds=0, terminal=False, win=False):
        if self.finished:
            raise RuntimeError('Reward requested after terminal outcome')
        if rounds < 0:
            raise ValueError('Completed round count cannot be negative')
        new_floors = max(0, floor-self.highest_floor)
        self.highest_floor = max(self.highest_floor, floor)
        progress = min(.25-self.floor_credit, new_floors*.25/60)
        cost = min(.10-self.round_cost, rounds*.001)
        self.floor_credit += progress
        self.round_cost += cost
        outcome = (1.0 if win else -1.0) if terminal else 0.0
        self.finished = terminal
        return progress-cost+outcome, {'floor_progress':progress,'round_cost':-cost,'outcome':outcome}


def completed_rounds(combat, previous_round, ended_turn):
    """An end-turn that kills the player still counts; selection confirmation does not."""
    return max(0, combat.turn_count-previous_round, int(ended_turn))
