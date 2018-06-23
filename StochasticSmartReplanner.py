import os
from collections import defaultdict
from functools import partial

import pickle
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
from pddlsim.local_simulator import LocalSimulator
import sys

from my_valid_actions_getter import MyValidActionsGetter
from utils import my_apply_action_to_state, encode_state


class StochasticSmartReplanner(Executor):
    def __init__(self, problem_path):
        super(StochasticSmartReplanner, self).__init__()
        self.problem_path = problem_path
        self.env_name = self.problem_path.split('-')[0]

        self.services = None

        self.plan = None
        self.is_off_plan = True
        self.steps_off_plan = None

        self.hash_visited_states = set()
        self.prev_state_hash = None
        self.prev_action = None
        self.uncompleted_goals = None
        self.active_goal = None

        # self.weights = defaultdict(partial(defaultdict, list))
        self.weights = defaultdict(float)
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))
        self.state_action_transition_count = defaultdict(partial(defaultdict, int))

        self.valid_actions_getter = None

    def initialize(self, services):
        self.services = services
        self.valid_actions_getter = MyValidActionsGetter(self.services.parser, self.services.perception)
        self.uncompleted_goals = self.services.goal_tracking.uncompleted_goals

        if os.path.exists(self.env_name + "_transitions"):
            self.transitions = self.load_obj(self.env_name + "_transitions")
        if os.path.exists(self.env_name + "_state_action_transition_count"):
            self.state_action_transition_count = self.load_obj(self.env_name + "_state_action_transition_count")
        if os.path.exists(self.problem_path + "_weights"):
            self.weights = self.load_obj(self.problem_path + "_weights")

    def next_action(self):
        state = self.services.perception.get_state()
        hash_state = encode_state(state)

        # check if done
        self.check_goals(state)
        if len(self.uncompleted_goals) == 0:
            self.save_obj(self.transitions, self.env_name + "_transitions")
            self.save_obj(self.state_action_transition_count, self.env_name + "_state_action_transition_count")
            self.save_obj(self.weights, self.problem_path + "_weights")
            return None

        # remember
        if hash_state not in self.hash_visited_states:
            self.hash_visited_states.add(encode_state(state))
        self.update(self.prev_state_hash, self.prev_action, hash_state)


        # choose
        applicable_actions = self.valid_actions_getter.get(state)
        possible_next_states = defaultdict(None)
        for applicable_action in applicable_actions:
            next_state = my_apply_action_to_state(state, applicable_action, self.services.parser)
            possible_next_states[applicable_action] = encode_state(next_state)

        actions_leading_to_not_seen_states = filter(lambda action_key: possible_next_states[action_key] not in self.hash_visited_states, possible_next_states)

        # todo: this is not a good option, but i dont think theres any choice here - backtrack?
        if len(actions_leading_to_not_seen_states) == 0:
            self.prev_state_hash = state
            self.prev_action = None
            return None

        if len(actions_leading_to_not_seen_states) == 1:
            self.prev_state_hash = hash_state
            self.prev_action = actions_leading_to_not_seen_states.pop(0)
            return self.prev_action

        if self.plan is None:
            self.make_plan()

        if len(self.plan) > 0:
            self.prev_action = self.plan.pop(0).lower()
            return self.prev_action





        return None

    def update(self, state_hash, action, next_state_hash):
        if state_hash is not None and action is not None:
            self.transitions[state_hash][action][next_state_hash] += 1
            self.state_action_transition_count[state_hash][action] += 1

    def save_obj(self, obj, name):
        pickle.dump(obj, open(name, 'w'))

    def load_obj(self, name):
        return pickle.load(open(name))

    def check_goals(self, state):
        for goal_condition in self.uncompleted_goals:
            if goal_condition.test(state):
                self.uncompleted_goals.remove(goal_condition)

        if self.active_goal is not None and self.active_goal.test(state):
            self.active_goal = None
            self.hash_visited_states = set()
            self.plan = None

    def make_plan(self):
        if self.active_goal is None:
            self.active_goal = self.uncompleted_goals[0]

        problem_path = self.services.problem_generator.generate_problem(
            self.active_goal, self.services.perception.get_state())
        self.plan = self.services.planner(self.services.pddl.domain_path, problem_path)

        for action in self.plan:
            self.weights[action] = 1







domain_path = "domain.pddl"
# problem_path = "t_5_5_5_multiple.pddl"
problem_path = "failing_actions_example.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path,
                                StochasticSmartReplanner(problem_path)))
