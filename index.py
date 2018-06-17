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
actions = parser.actions
goal_states = get_goal_states(parser.goals, states)

distances_from_goals = get_distances_from_goals(goal_states, states, valid_actions_getter, parser)



print(LocalSimulator(local).run(domain_path, problem_path, My_Executer()))