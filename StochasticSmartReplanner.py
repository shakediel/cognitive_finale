import os
from collections import defaultdict
from functools import partial

import pickle
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
from pddlsim.local_simulator import LocalSimulator
import sys

from my_valid_actions_getter import MyValidActionsGetter
from utils import my_apply_action_to_state, make_nested_hash


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

        self.weights = defaultdict(partial(defaultdict, list))
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
        if os.path.exists(self.problem_path + "_weights"):
            self.weights = self.load_obj(self.problem_path + "_weights")

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            self.save_obj(self.transitions, self.env_name + "_transitions")
            self.save_obj(self.state_action_transition_count, self.env_name + "_state_action_transition_count")
            self.save_obj(self.weights, self.problem_path + "_weights")
            return None

        if self.plan is not None:
            if len(self.plan) > 0:
                return self.plan.pop(0).lower()
            return None

        # remember
        state = self.services.perception.get_state()
        if state not in self.visited_states:
            self.visited_states.append(state)
        self.update(self.prev_state, self.prev_action, state)


        # choose
        applicable_actions = self.valid_actions_getter.get(state)
        possible_next_states = defaultdict(None)
        for applicable_action in applicable_actions:
            next_state = my_apply_action_to_state(state, applicable_action, self.services.parser)
            possible_next_states[applicable_action] = next_state

        not_seen_states_actions = filter(lambda action_key: possible_next_states[action_key] not in self.visited_states, possible_next_states)

        if len(not_seen_states_actions) == 0:
            self.prev_state = state
            self.prev_action = None
            return None

        if len(not_seen_states_actions) == 1:
            self.prev_state = state
            self.prev_action = not_seen_states_actions.pop(0)
            return self.prev_action



        problem_path = self.services.problem_generator.generate_problem(
            self.services.goal_tracking.uncompleted_goals[0], self.services.perception.get_state())
        self.plan = self.services.planner(self.services.pddl.domain_path, problem_path)

        if len(self.plan) > 0:
            return self.plan.pop(0).lower()
        return None

    def update(self, state, action, next_state):
        if state is not None and action is not None:
            state_index = self.states.index(state)
            next_state_index = self.states.index(next_state)

            self.transitions[state_index][action][next_state_index] += 1
            self.state_action_transition_count[state_index][action] += 1

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
print(LocalSimulator(local).run(domain_path, problem_path,
                                StochasticSmartReplanner(problem_path)))
