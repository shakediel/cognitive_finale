from collections import defaultdict
from functools import partial

from pddlsim.executors.executor import Executor


class My_Executer(Executor):
    def __init__(self):
        super(My_Executer, self).__init__()
        self.services = None

        self.rewards = defaultdict(partial(defaultdict, list))
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))  # S --> A --> S' --> counts
        self.r_s_a_counts = defaultdict(partial(defaultdict, int))  # S --> A --> #rs
        self.t_s_a_counts = defaultdict(partial(defaultdict, int))  # S --> A --> #ts
        self.prev_state = None
        self.prev_action = None

    def initialize(self, services):
        self.services = services

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            #todo: add save
            return None

        state = self.services.perception.get_state()
        t=9
