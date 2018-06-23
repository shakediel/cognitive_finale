import copy
import os
import random
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
        self.off_plan_punish_factor = 0.1
        self.lookahead = 4
        self.gamma = 0.9

        self.hash_visited_states = set()
        self.prev_state_hash = None
        self.prev_action = None
        self.uncompleted_goals = None
        self.active_goal = None

        self.weights = defaultdict(partial(defaultdict, partial(defaultdict, int)))
        # self.weights = defaultdict(float)
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
        state_hash = encode_state(state)

        # check if done
        self.check_goals(state)
        if len(self.uncompleted_goals) == 0:
            self.save_obj(self.transitions, self.env_name + "_transitions")
            self.save_obj(self.state_action_transition_count, self.env_name + "_state_action_transition_count")
            self.save_obj(self.weights, self.problem_path + "_weights")
            return None

        # remember
        if state_hash not in self.hash_visited_states:
            self.hash_visited_states.add(encode_state(state))
        self.update(self.prev_state_hash, self.prev_action, state_hash)


        # choose
        applicable_actions = self.valid_actions_getter.get(state)

        if self.plan is not None and len(self.plan) > 0:
            # if state_hash == self.prev_state_hash:
            action = self.choose(state)

            t=9
            self.prev_action = action
            self.prev_state_hash = state_hash
            return self.prev_action


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
            self.prev_state_hash = state_hash
            self.prev_action = actions_leading_to_not_seen_states.pop(0)
            return self.prev_action

        if self.plan is None:
            self.make_plan(state)
            self.prev_state_hash = state_hash
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
            self.weights = defaultdict(partial(defaultdict, partial(defaultdict, int)))

    def make_plan(self, state):
        curr_state = copy.deepcopy(state)
        if self.active_goal is None:
            self.active_goal = self.uncompleted_goals[0]

        problem = self.services.problem_generator.generate_problem(self.active_goal, curr_state)
        self.plan = self.services.planner(self.services.pddl.domain_path, problem)

        for action in self.plan:
            curr_state_hash = encode_state(curr_state)
            expected_next_state = my_apply_action_to_state(curr_state, action, self.services.parser)
            self.weights[curr_state_hash][action.lower()][encode_state(expected_next_state)] = 1
            curr_state = expected_next_state

    def choose(self, state):
        action = self.get_max_q_action(state, self.lookahead)
        return action

    def get_max_q_action(self, state, lookahead):
        # expected_next_state = my_apply_action_to_state(self.prev_state)
        prev_action_weight = max(self.weights[self.prev_state_hash][self.prev_action].values())
        return self._compute_max_qval_action_pair(state, lookahead, prev_action_weight)[1]

    def _compute_max_qval_action_pair(self, state, lookahead, prev_action_weight):
        state_hash = encode_state(state)
        predicted_returns = defaultdict(float)
        actions = self.valid_actions_getter.get(state)
        for action in actions:
            # expansion...
            # todo: only expand if next expected state was not seen
            edge_weight = prev_action_weight * self.off_plan_punish_factor
            expected_next_state = my_apply_action_to_state(state, action, self.services.parser)
            expected_next_state_hash = encode_state(expected_next_state)
            if self.weights[state_hash][action][expected_next_state_hash] == 0 or self.weights[state_hash][action][expected_next_state_hash] < edge_weight:
                self.weights[state_hash][action][expected_next_state_hash] = edge_weight

            q_s_a = self.get_q_value(state, action, lookahead)
            predicted_returns[action] = q_s_a

        max_q_val = max(predicted_returns.values())
        best_actions = list()
        for action_name in predicted_returns:
            if predicted_returns[action_name] == max_q_val:
                best_actions.append(action_name)

        best_action = random.choice(best_actions)

        return max_q_val, best_action

    def get_q_value(self, state, action, lookahead):
        if lookahead <= 0 or len(self.valid_actions_getter.get(state)) == 0:
            return self._get_reward(state, action)

        expected_future_return = self.gamma * self._compute_expected_future_return(state, action, lookahead)
        q_val = self._get_reward(state, action) + expected_future_return

        return q_val

    def _compute_expected_future_return(self, state, action, lookahead):
        state_hash = encode_state(state)
        next_action_state_occurence = self.state_action_transition_count[state_hash][action]

        next_state_occurence_dict = self.transitions[state_hash][action]
        state_probabilities = defaultdict(float)

        for next_state_hash in next_state_occurence_dict:
            count = next_state_occurence_dict[next_state_hash]
            # if count == 0:
            #     state_probabilities[next_state_hash] = 1
            # else:
            state_probabilities[next_state_hash] = (count / next_action_state_occurence)

        weighted_future_returns = list()
        for state_hash in state_probabilities:
            prev_action_weight = max(self.weights[state_hash][action].values())
            next_state = my_apply_action_to_state(state, action, self.services.parser)
            weighted_future_returns.append(self.get_max_q_value(next_state, lookahead - 1, prev_action_weight) * state_probabilities[state_hash])

        return sum(weighted_future_returns)

    def get_max_q_value(self, state, lookahead, prev_action_weight):
        return self._compute_max_qval_action_pair(state, lookahead, prev_action_weight)[0]

    def _get_reward(self, state, action):
        state_hash = encode_state(state)
        reward = max(self.weights[state_hash][action].values())
        return reward


domain_path = "domain.pddl"
# problem_path = "t_5_5_5_multiple.pddl"
problem_path = "ahinoam_problem.pddl"
# problem_path = "failing_actions_example.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path,
                                StochasticSmartReplanner(problem_path)))
