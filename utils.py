import copy


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


def get_all_states(root_state):
    pass