from collections import defaultdict

from pddlsim.local_simulator import LocalSimulator
from pddlsim.planner import local
from pddlsim.services.perception import Perception
from pddlsim.fd_parser import FDParser
from pddlsim.simulator import Simulator

from my_valid_actions_getter import MyValidActionsGetter
from r_max import My_Executer
from utils import get_all_states, get_goal_states, get_distances_from_goals

domain_path = "domain.pddl"
problem_path = "t_5_5_5_multiple.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]

parser = FDParser(domain_path, problem_path)
sim = Simulator(parser)
perception = Perception(sim.perceive_state)
valid_actions_getter = MyValidActionsGetter(parser, perception)

root_state = perception.get_state()

states = get_all_states(root_state, valid_actions_getter, parser)
actions = [action for action in parser.actions]
goal_states = get_goal_states(parser.goals, states)

distances_from_goals = get_distances_from_goals(goal_states, states, valid_actions_getter, parser)
state_discovery_reward = defaultdict(float)
for state_index in distances_from_goals:
    state_discovery_reward[state_index] = 0
    state_distances = distances_from_goals[state_index]
    for goal_index in state_distances:
        state_discovery_reward[state_index] += 1. / state_distances[goal_index]

max_reward = len(goal_states) * max(state_discovery_reward.values())
state_recurrence_punish = min(state_discovery_reward.values())
bad_action_punish = max(state_discovery_reward.values())
lookahead = 4
known_threshold = len(states) * len(actions)

print(LocalSimulator(local).run(domain_path, problem_path,
    My_Executer(problem_path, states, actions, goal_states, state_discovery_reward, max_reward, state_recurrence_punish,
                bad_action_punish, lookahead, known_threshold)))