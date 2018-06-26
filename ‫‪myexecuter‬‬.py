import sys

from pddlsim.local_simulator import LocalSimulator
from pddlsim.planner import local

from StochasticSmartReplanner import StochasticSmartReplanner

domain_path = "domain.pddl"
problem_path = "t_5_5_5_multiple.pddl"
# problem_path = "ahinoam_problem.pddl"
# problem_path = "failing_actions_example.pddl"
# domain_path = "freecell_domain.pddl"
# problem_path = "freecell_problem.pddl"
# domain_path = "rover_domain.pddl"
# problem_path = "rover_problem.pddl"
# domain_path = "satellite_domain.pddl"
# problem_path = "satellite_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path,
                                StochasticSmartReplanner(problem_path)))
