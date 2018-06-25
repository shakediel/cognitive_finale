import copy
from collections import defaultdict
from collections import deque
from functools import partial
import hashlib
import json

import pickle


def my_apply_action_to_state(orig_state, action, parser):
    state = copy.deepcopy(orig_state)
    action_name, param_names = parser.parse_action(action)

    action = parser.actions[action_name]
    params = map(parser.get_object, param_names)

    param_mapping = action.get_param_mapping(params)

    for (predicate_name, entry) in action.to_delete(param_mapping):
        predicate_set = state[predicate_name]
        if entry in predicate_set:
            predicate_set.remove(entry)

    for (predicate_name, entry) in action.to_add(param_mapping):
        state[predicate_name].add(entry)

    return state


def get_all_states(root_state, valid_actions_getter, parser):
    states = list()
    queue = deque()
    queue.append(root_state)
    states.append(root_state)

    while queue:
        curr_state = queue.popleft()
        curr_state_valid_actions = valid_actions_getter.get(curr_state)
        for action in curr_state_valid_actions:
            next_state = my_apply_action_to_state(curr_state, action, parser)
            if next_state not in states:
                states.append(next_state)
                queue.append(next_state)

    return states


def get_goal_states(goal_conditions, states):
    goal_states = list()
    for goal_condition in goal_conditions:
        for state in states:
            if goal_condition.test(state):
                goal_states.append(state)
    return goal_states


def get_distances_from_goals(goal_states, states, valid_actions_getter, parser):
    goal_distances = defaultdict(partial(defaultdict, int))
    for goal_state in goal_states:
        goal_state_index = states.index(goal_state)
        goal_distances[goal_state_index][goal_state_index] = 1

        queue = deque()
        queue.append(goal_state)
        visited_states = list()
        visited_states.append(goal_state)
        while queue:
            curr_state = queue.popleft()
            curr_state_valid_actions = valid_actions_getter.get(curr_state)
            curr_state_index = states.index(curr_state)
            for action in curr_state_valid_actions:
                next_state = my_apply_action_to_state(curr_state, action, parser)
                if next_state not in visited_states:
                    queue.append(next_state)
                    visited_states.append(next_state)
                    goal_distances[states.index(next_state)][goal_state_index] = goal_distances[curr_state_index][goal_state_index] + 1

    return goal_distances


def encode_state(state):
    copied_state = copy.deepcopy(state)
    for key in state:
        val = copied_state[key]
        if isinstance(val, (set)):
            copied_state[key] = list(val)
            copied_state[key].sort()

    json_representation = json.dumps(copied_state, sort_keys=True)
    return hashlib.sha1(json_representation).hexdigest()


def median(numbers):
    numbers = sorted(numbers)
    center = len(numbers) / 2
    if len(numbers) % 2 == 0:
        return sum(numbers[center - 1:center + 1]) / 2.0
    else:
        return numbers[center]


def save_obj(obj, name):
    pickle.dump(obj, open(name, 'w'))


def load_obj(name):
    return pickle.load(open(name))
