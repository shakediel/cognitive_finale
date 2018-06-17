from pddlsim.executors.executor import Executor
from pddlsim.planner import local
from pddlsim.local_simulator import LocalSimulator
import sys

class SmartReplanner(Executor):
    def __init__(self):
        super(SmartReplanner, self).__init__()
        self.services = None
        self.plan = None

    def initialize(self, services):
        self.services = services

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            return None

        if self.plan is not None:
            if len(self.plan) > 0:
                return self.plan.pop(0).lower()
            return None

        options = self.services.valid_actions.get()


        if len(options) == 0:
            return None

        if len(options) == 1:
            return options[0]

        problem_path = self.services.problem_generator.generate_problem(
            self.services.goal_tracking.uncompleted_goals[0], self.services.perception.get_state())
        self.plan = self.services.planner(self.services.pddl.domain_path, problem_path)

        if len(self.plan) > 0:
            return self.plan.pop(0).lower()
        return None



# domain_path = "../ex1/domain.pddl"
# problem_path = "../ex1/problem.pddl"
domain_path = "freecell_domain.pddl"
problem_path = "freecell_problem.pddl"
# domain_path = sys.argv[1]
# problem_path = sys.argv[2]
print(LocalSimulator(local).run(domain_path, problem_path, SmartReplanner()))
