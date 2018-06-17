import os
import random
from collections import defaultdict
from functools import partial

import pickle
from pddlsim.executors.executor import Executor

from my_valid_actions_getter import MyValidActionsGetter


class My_Executer(Executor):
    def __init__(self, problem_path, states, actions, goal_states, state_discovery_reward, max_reward, state_recurrence_punish, bad_action_punish, lookahead, known_threshold):
        super(My_Executer, self).__init__()
        self.services = None
        self.time = 0
        self.route = set()

        self.problem_path = problem_path
        self.env_name = self.problem_path.split('-')[0]

        self.states = states
        self.actions = actions
        self.goal_states = goal_states
        self.state_discovery_reward = state_discovery_reward
        self.max_reward = max_reward
        self.state_recurrence_punish = state_recurrence_punish
        self.bad_action_punish = bad_action_punish
        self.lookahead = lookahead
        self.known_threshold = known_threshold
        self.gamma = 0.95

        self.visited_states = list()
        self.prev_state = None
        self.prev_action = None
        self.prev_state_valid_actions = None

        self.rewards = defaultdict(partial(defaultdict, list))
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))
        self.state_action_rewards_count = defaultdict(partial(defaultdict, int))
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
        if os.path.exists(self.problem_path + "_state_action_rewards_count"):
            self.state_action_rewards_count = self.load_obj(self.problem_path + "_state_action_rewards_count")

        num_known_sa = self.get_num_known_sa()
        if num_known_sa == len(self.states) * len(self.actions):
            t=9

    #todo: limit num of iterations??
    def next_action(self, state=None):
        if self.services.goal_tracking.reached_all_goals() or self.time >= len(self.states) * len(self.actions) * len(self.actions):
            self.reward_route(self.services.goal_tracking.reached_all_goals())
            self.save_obj(self.transitions, self.env_name + "_transitions")
            self.save_obj(self.state_action_transition_count, self.env_name + "_state_action_transition_count")
            self.save_obj(self.rewards, self.problem_path + "_rewards")
            self.save_obj(self.state_action_rewards_count, self.problem_path + "_state_action_rewards_count")
            return None

        self.time += 1

        if state is None:
            state = self.services.perception.get_state()

        reward = 0
        if self.prev_state_valid_actions is not None:
            if not any(self.prev_action in action for action in self.prev_state_valid_actions):
                reward -= self.bad_action_punish
        else:
            if state in self.visited_states:
                reward -= self.state_recurrence_punish
            else:
                reward += self.state_discovery_reward[self.states.index(state)]
                self.visited_states.append(state)

        action_name = self.choose(state, reward)

        self.prev_state = state
        self.prev_action = action_name
        self.prev_state_valid_actions = self.valid_actions_getter.get(state)

        if not any(action_name in action for action in self.prev_state_valid_actions):
            return self.next_action(state)

        action = next(action for action in self.prev_state_valid_actions if action_name in action)

        self.route.add(tuple([self.states.index(state), action_name]))

        return action

    def choose(self, state, reward):
        self.update(self.prev_state, self.prev_action, reward, state)

        action = self.get_max_q_action(state, self.lookahead)

        self.prev_action = action
        self.prev_state = state

        return action

    def update(self, state, action, reward, next_state):
        if state is not None and action is not None:
            state_index = self.states.index(state)
            next_state_index = self.states.index(next_state)

            self.rewards[state_index][action] += [reward]
            self.state_action_rewards_count[state_index][action] += 1

            self.transitions[state_index][action][next_state_index] += 1
            self.state_action_transition_count[state_index][action] += 1

    def get_max_q_action(self, state, lookahead):
        return self._compute_max_qval_action_pair(state, lookahead)[1]

    def get_max_q_value(self, state, lookahead):
        return self._compute_max_qval_action_pair(state, lookahead)[0]

    def _compute_max_qval_action_pair(self, state, lookahead):
        predicted_returns = defaultdict(float)
        for action in self.actions:
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
        if lookahead <= 0 or len(self.valid_actions_getter.get(state)) is 0:
            return self._get_reward(state, action)

        expected_future_return = self.gamma * self._compute_exp_future_return(state, action, lookahead)
        q_val = self._get_reward(state, action) + expected_future_return

        return q_val

    def _compute_exp_future_return(self, state, action, lookahead):
        state_index = self.states.index(state)
        next_action_state_occurence = self.state_action_transition_count[state_index][action]

        next_state_occurence_dict = self.transitions[self.states.index(state)][action]
        state_weights = defaultdict(float)
        if next_action_state_occurence >= self.known_threshold:
            normal = float(sum(next_state_occurence_dict.values()))
            for next_state in next_state_occurence_dict:
                count = next_state_occurence_dict[next_state]
                state_weights[next_state] = (count / normal)
        else:
            for next_state in next_state_occurence_dict:
                state_weights[next_state] = 1

        weighted_future_returns = list()
        for state_index in state_weights:
            weighted_future_returns.append(self.get_max_q_value(self.states[state_index], lookahead - 1) * state_weights[state_index])

        return sum(weighted_future_returns)

    def _get_reward(self, state, action):
        state_index = self.states.index(state)
        if self.state_action_rewards_count[state_index][action] >= self.known_threshold:
            rewards_s_a = self.rewards[state_index][action]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            return self.max_reward

    def save_obj(self, obj, name):
        pickle.dump(obj, open(name, 'w'))

    def load_obj(self, name):
        return pickle.load(open(name))

    def is_known(self, s, a):
        return self.state_action_transition_count[s][a] >= self.known_threshold

    def get_num_known_sa(self):
        count = 0
        for state in self.states:
            for action_name in self.actions:
                if self.state_action_transition_count[self.states.index(state)][action_name] >= self.known_threshold:
                    count += 1
        return count

    def reward_route(self, reached_all_goals):
        if not reached_all_goals:
            return

        for state_action_tuple in self.route:
            self.rewards[state_action_tuple[0]][state_action_tuple[1]] += [self.max_reward]



