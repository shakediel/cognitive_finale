import os
from collections import defaultdict
from functools import partial

import pickle
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
from pddlsim.local_simulator import LocalSimulator
import sys

from my_valid_actions_getter import MyValidActionsGetter


class StochasticSmartReplanner(Executor):
    def __init__(self, problem_path):
        super(StochasticSmartReplanner, self).__init__()
        self.problem_path = problem_path
        self.env_name = self.problem_path.split('-')[0]

        self.services = None
        self.plan = None

        self.visited_states = list()
        self.prev_state = None
        self.prev_action = None

        self.rewards = defaultdict(partial(defaultdict, list))
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))
        self.state_action_transition_count = defaultdict(partial(defaultdict, int))

        self.valid_actions_getter = None

    def initialize(self, services):
        self.services = services
        self.valid_actions_getter = MyValidActionsGetter(self.services.parser, self.services.perception)

        if os.path.exists(self.env_name + "_transitions"):
            self.transitions = self.load_obj(self.env_name + "_transitions")
        if os.path.exists(self.env_name + "_state_action_transition_count"):
            self.state_action_transition_count = self.load_obj(self.env_name + "_state_action_transition_count")
        if os.path.exists(self.problem_path + "_rewards"):
            self.rewards = self.load_obj(self.problem_path + "_rewards")

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            return None

        if self.plan is not None:
            if len(self.plan) > 0:
                return self.plan.pop(0).lower()
            return None

        options = self.services.valid_actions.get()

        if len(options) == 0:
            return None

        if len(options) == 1:
            return options[0]

        problem_path = self.services.problem_generator.generate_problem(
            self.services.goal_tracking.uncompleted_goals[0], self.services.perception.get_state())
        self.plan = self.services.planner(self.services.pddl.domain_path, problem_path)

        if len(self.plan) > 0:
            return self.plan.pop(0).lower()
        return None

    def save_obj(self, obj, name):
        pickle.dump(obj, open(name, 'w'))

    def load_obj(self, name):
        return pickle.load(open(name))


domain_path = "domain.pddl"
# problem_path = "t_5_5_5_multiple.pddl"
problem_path = "failing_actions_example.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path, StochasticSmartReplanner()))