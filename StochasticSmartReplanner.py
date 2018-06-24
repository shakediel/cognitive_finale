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
from utils import my_apply_action_to_state, encode_state, save_obj, load_obj


class StochasticSmartReplanner(Executor):
    def __init__(self, problem_name):
        super(StochasticSmartReplanner, self).__init__()
        self.problem_path = problem_name
        self.env_name = self.problem_path.split('-')[0]

        self.services = None

        self.plan = None
        self.is_off_plan = True
        self.off_plan_punish_factor = 0.1
        self.state_recurrence_punish = 0.1
        self.lookahead = 3
        self.gamma = 0.9
        self.known_threshold = 1
        self.last_in_plan_transition_weight = 0

        self.visited_states_hash = set()
        self.prev_state_hash = None
        self.prev_action = None
        self.uncompleted_goals = None
        self.active_goal = None

        self.weights = defaultdict(partial(defaultdict, int))
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))
        self.state_action_transition_count = defaultdict(partial(defaultdict, int))
        self.rewards = defaultdict(partial(defaultdict, list))
        self.state_action_rewards_count = defaultdict(partial(defaultdict, int))

        self.valid_actions_getter = None

    def initialize(self, services):
        self.services = services
        self.valid_actions_getter = MyValidActionsGetter(self.services.parser, self.services.perception)
        self.uncompleted_goals = self.services.goal_tracking.uncompleted_goals

        if os.path.exists(self.env_name + "_transitions"):
            self.transitions = load_obj(self.env_name + "_transitions")
        if os.path.exists(self.env_name + "_state_action_transition_count"):
            self.state_action_transition_count = load_obj(self.env_name + "_state_action_transition_count")

    def next_action(self):
        state = self.services.perception.get_state()
        state_hash = encode_state(state)

        # check if done
        self.check_goals(state)
        if len(self.uncompleted_goals) == 0:
            save_obj(self.transitions, self.env_name + "_transitions")
            save_obj(self.state_action_transition_count, self.env_name + "_state_action_transition_count")
            return None

        # remember
        self.update(self.prev_state_hash, self.prev_action, state_hash)

        # choose
        if self.plan is not None:
            if self.prev_action.upper() not in self.plan and\
                            self.weights[self.prev_state_hash][self.prev_action] <= self.last_in_plan_transition_weight * self.off_plan_punish_factor ** self.lookahead:
                self.plan = None

        if self.plan is not None:
            if self.prev_action.upper() not in self.plan and\
                            self.weights[self.prev_state_hash][self.prev_action] <= self.last_in_plan_transition_weight * self.off_plan_punish_factor ** self.lookahead:
                self.make_plan(state)

            action = self.choose(state)

            self.prev_action = action
            self.prev_state_hash = state_hash
            return self.prev_action

        applicable_actions = self.valid_actions_getter.get(state)
        possible_next_states = defaultdict(None)
        for applicable_action in applicable_actions:
            next_state = my_apply_action_to_state(state, applicable_action, self.services.parser)
            possible_next_states[applicable_action] = encode_state(next_state)

        actions_leading_to_not_seen_states = filter(lambda action_key: possible_next_states[action_key] not in self.visited_states_hash, possible_next_states)

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

            action = self.choose(state)

            self.prev_state_hash = state_hash
            self.prev_action = action
            return self.prev_action

        return None

    def update(self, state_hash, action, next_state_hash):
        if state_hash is not None and action is not None:
            if self.plan is not None and action.upper() in self.plan:
                self.last_in_plan_transition_weight = self.weights[state_hash][action]

            reward = 0
            if next_state_hash in self.visited_states_hash:
                reward -= self.state_recurrence_punish
            else:
                reward += self.weights[state_hash][action]
                self.visited_states_hash.add(state_hash)

            self.rewards[state_hash][action] += [reward]
            self.transitions[state_hash][action][next_state_hash] += 1
            self.state_action_transition_count[state_hash][action] += 1
            self.state_action_rewards_count[state_hash][action] += 1

    def check_goals(self, state):
        for goal_condition in self.uncompleted_goals:
            if goal_condition.test(state):
                self.uncompleted_goals.remove(goal_condition)

        if self.active_goal is not None and self.active_goal.test(state):
            self.active_goal = None
            self.visited_states_hash = set()
            self.plan = None
            self.weights = defaultdict(partial(defaultdict, int))

    def make_plan(self, state):
        curr_state = copy.deepcopy(state)
        if self.active_goal is None:
            self.active_goal = self.uncompleted_goals[0]

        problem = self.services.problem_generator.generate_problem(self.active_goal, curr_state)
        self.plan = self.services.planner(self.services.pddl.domain_path, problem)
        self.last_in_plan_transition_weight = 0

        for i in range(len(self.plan)):
            action = self.plan[i]
            curr_state_hash = encode_state(curr_state)
            weight = float(i + 1) / len(self.plan)
            if self.weights[curr_state_hash][action.lower()] < weight:
                self.weights[curr_state_hash][action.lower()] = weight
            curr_state = my_apply_action_to_state(curr_state, action, self.services.parser)

    def choose(self, state):
        action = self.get_max_q_action(state, self.lookahead)
        return action

    def get_max_q_action(self, state, lookahead):
        prev_action_weight = self.weights[self.prev_state_hash][self.prev_action]
        return self._compute_max_qval_action_pair(state, lookahead, prev_action_weight)[1]

    def _compute_max_qval_action_pair(self, state, lookahead, prev_action_weight):
        state_hash = encode_state(state)
        predicted_returns = defaultdict(float)
        actions = self.valid_actions_getter.get(state)
        for action in actions:
            # expansion...
            edge_weight = prev_action_weight * self.off_plan_punish_factor
            if self.weights[state_hash][action] == 0 or self.weights[state_hash][action] < edge_weight:
                self.weights[state_hash][action] = edge_weight

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
        next_action_state_occurrence = self.state_action_transition_count[state_hash][action]

        next_state_occurrence_dict = self.transitions[state_hash][action]
        state_probabilities = defaultdict(float)
        for next_state_hash in next_state_occurrence_dict:
            count = next_state_occurrence_dict[next_state_hash]
            if count < 5:
                state_probabilities[next_state_hash] = 1
            else:
                state_probabilities[next_state_hash] = (count / next_action_state_occurrence)

        weighted_future_returns = list()
        for next_state_hash in state_probabilities:
            prev_action_weight = self.weights[state_hash][action]
            next_state = my_apply_action_to_state(state, action, self.services.parser)
            weighted_future_returns.append(self.get_max_q_value(next_state, lookahead - 1, prev_action_weight) * state_probabilities[next_state_hash])

        return sum(weighted_future_returns)

    def get_max_q_value(self, state, lookahead, prev_action_weight):
        return self._compute_max_qval_action_pair(state, lookahead, prev_action_weight)[0]

    def _get_reward(self, state, action):
        state_hash = encode_state(state)
        if self.state_action_rewards_count[state_hash][action] >= self.known_threshold:
            state_action_rewards = self.rewards[state_hash][action]
            reward = float(sum(state_action_rewards)) / len(state_action_rewards)
        else:
            # for exploration
            reward = 1
        return reward


# domain_path = "domain.pddl"
# problem_path = "t_5_5_5_multiple.pddl"
# problem_path = "ahinoam_problem.pddl"
# problem_path = "failing_actions_example.pddl"
domain_path = "freecell_domain.pddl"
problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path,
                                StochasticSmartReplanner(problem_path)))
