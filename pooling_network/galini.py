# Copyright 2020 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pyomo.environ as pe

from galini.branch_and_bound.node import NodeSolution
from galini.config import CutsGeneratorOptions, BoolOption
from galini.cuts import CutsGenerator
from galini.pyomo.util import update_solver_options
from galini.solvers.solution import load_solution_from_model

from pooling_network.cuts import generate_valid_cuts
from pooling_network.heuristic import mip_heuristic, derive_fractional_flow_variables
from pooling_network.iterators import pooling_data_objects


class PoolingProblemCutsGenerator(CutsGenerator):
    name = 'pooling'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini
        self.logger = galini.get_logger(__name__)

        self.root_node_only = config['root_node_only']
        self._pooling = None
        self._at_root_node = False

    @staticmethod
    def cuts_generator_options():
        return CutsGeneratorOptions(
            PoolingProblemCutsGenerator.name, [
                BoolOption('root_node_only', default=False, description='Generate cuts at the root node only'),
        ])

    def before_start_at_root(self, problem, relaxed_problem):
        for pooling in pooling_data_objects(relaxed_problem, descend_into=True, active=True):
            if self._pooling is not None:
                raise RuntimeError('Only support one pooling problem')
            self._pooling = pooling

        if self._pooling is not None:
            self._at_root_node = True
            self._pooling.add_inequalities()

    def after_end_at_root(self, problem, relaxed_problem, solution):
        self._at_root_node = False

    def before_start_at_node(self, problem, relaxed_problem):
        self._at_root_node = False

    def after_end_at_node(self, problem, relaxed_problem, solution):
        pass

    def has_converged(self, state):
        if self.root_node_only:
            return not self._at_root_node

        if not self._pooling:
            return True

        return False

    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        if self.root_node_only and not self._at_root_node:
            return
        generator = generate_valid_cuts(self._pooling.inequalities, self._pooling, self._pooling.pooling_problem)

        if not self.galini.paranoid_mode:
            cuts = [c for c, _ in generator]
            return cuts

        cuts = []
        for i, (cut, cut_info) in enumerate(generator):
            self.logger.info('Cut {}: {}', i, cut_info)
            cuts.append(cut)
        return cuts


class PoolingPrimalSearchStrategy:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.logger = self.algorithm.galini.get_logger(__name__)

    def solve(self, model, tree, node):
        pooling = None
        for pooling in pooling_data_objects(model, descend_into=True, active=True):
            break
        self.logger.info('PoolingPrimalSearchStrategy with problem: {}', pooling)
        if pooling is None:
            self.logger.info('Exit: no pooling problem detected')
            return None

        problem = pooling.pooling_problem

        with mip_heuristic(model.pooling, problem, tau=1):
            update_solver_options(self.algorithm._mip_solver, timelimit=60, relative_gap=0.01)
            results = self.algorithm._mip_solver.solve(model, tee=True)
            # Make sure we drop extra variables.

        derive_fractional_flow_variables(model.pooling)

        model.pooling.v.fix()
        model.pooling.y.fix()
        model.pooling.z.fix()
        model.pooling.q.fix()

        results = self.algorithm._mip_solver.solve(model, tee=True)
        solution = load_solution_from_model(results, model)

        model.pooling.v.unfix()
        model.pooling.y.unfix()
        model.pooling.z.unfix()
        model.pooling.q.unfix()

        self.logger.info('Pooling heuristic solution: {}', solution)
        if solution is not None:
            return NodeSolution(None, solution)

        return None
