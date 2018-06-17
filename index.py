from pddlsim.local_simulator import LocalSimulator
from pddlsim.planner import local
from pddlsim.services.perception import Perception
from pddlsim.fd_parser import FDParser
from pddlsim.simulator import Simulator

from my_valid_actions_getter import MyValidActionsGetter
from r_max import My_Executer
from utils import get_all_states

domain_path = "domain.pddl"
problem_path = "t_5_5_5_multiple.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]

parser = FDParser(domain_path, problem_path)
sim = Simulator(parser)
perception = Perception(sim.perceive_state)
my_valid_actions_getter = MyValidActionsGetter(parser, perception)

root_state = perception.get_state()
valid_action = my_valid_actions_getter.get(root_state)



states = get_all_states(None)




print(LocalSimulator(local).run(domain_path, problem_path, My_Executer()))