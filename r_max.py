import random
from collections import defaultdict
from functools import partial

from pddlsim.executors.executor import Executor

from my_valid_actions_getter import MyValidActionsGetter


class My_Executer(Executor):
    def __init__(self, problem_path, states, actions, goal_states, state_discovery_reward, max_reward, state_recurrence_punish, bad_action_punish, lookahead, known_threshold):
        super(My_Executer, self).__init__()
        self.services = None

        self.problem_path = problem_path
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
        self.transitions = defaultdict(partial(defaultdict, partial(defaultdict, int)))  # S --> A --> S' --> counts
        self.r_s_a_counts = defaultdict(partial(defaultdict, int))  # S --> A --> #rs
        self.t_s_a_counts = defaultdict(partial(defaultdict, int))  # S --> A --> #ts

        self.valid_actions_getter = None

    def initialize(self, services):
        self.services = services
        self.valid_actions_getter = MyValidActionsGetter(self.services.parser, self.services.perception)

        #todo: add load

    #todo: limit num of iterations??
    def next_action(self, state=None):
        if self.services.goal_tracking.reached_all_goals():
            #todo: add save
            return None

        if state is None:
            state = self.services.perception.get_state()

        reward = 0
        if self.prev_state_valid_actions is not None:
            if not any(self.prev_action in action for action in self.prev_state_valid_actions):
                reward += self.bad_action_punish
        else:
            if state in self.visited_states:
                reward -= self.state_recurrence_punish
            else:
                reward += self.state_discovery_reward[self.states.index(state)]
                self.visited_states.append(state)

        action_name = self.choose(state, reward)



        t=9

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

            if self.r_s_a_counts[state_index][action] <= self.known_threshold:
                self.rewards[state_index][action] += [reward]
                self.r_s_a_counts[state_index][action] += 1

            if self.t_s_a_counts[state_index][action] <= self.known_threshold:
                self.transitions[state_index][action][next_state_index] += 1
                self.t_s_a_counts[state_index][action] += 1

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
        next_action_state_occurence = self.t_s_a_counts[state_index][action]

        state_weights = defaultdict(float)
        if next_action_state_occurence >= self.known_threshold:
            next_state_occurence_dict = self.transitions[self.states.index(state)][action]
            normal = float(sum(next_state_occurence_dict.values()))
            for next_state in next_state_occurence_dict.keys():
                count = next_state_occurence_dict[next_state]
                state_weights[next_state] = (count / normal)
        else:
            for next_state in self.states:
                state_weights[self.states.index(next_state)] = 1

        weighted_future_returns = list()
        for state_index in state_weights:
            weighted_future_returns.append(self.get_max_q_value(self.states[state_index], lookahead - 1) * state_weights[state_index])

        return sum(weighted_future_returns)

    def _get_reward(self, state, action):
        state_index = self.states.index(state)
        if self.r_s_a_counts[state_index][action] >= self.known_threshold:
            rewards_s_a = self.rewards[state_index][action]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            return self.max_reward
