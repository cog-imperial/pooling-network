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
from typing import Optional

import pyomo.environ as pyo
from pyomo.core.base.block import declare_custom_block, _BlockData

from pooling_network.cuts import add_valid_cuts
from pooling_network.inequalities import add_all_pooling_inequalities
from pooling_network.network import Network
from pooling_network.pq import pooling_problem_pq_formulation, minimize_cost_objective, minimize_flow_cost_objective


@declare_custom_block(name='PoolingPQFormulation')
class PoolingPQFormulationData(_BlockData):
    def __init__(self, component):
        super().__init__(component)

        self._built = False
        self.pooling_problem: Optional[Network] = None
        self.inequalities = None

    def set_pooling_problem(self, problem: Network):
        self.pooling_problem = problem

    def rebuild(self, skip_product_quality=False):
        if self.pooling_problem is None:
            raise RuntimeError("Must set pooling_problem")

        self._built = True
        pooling_problem_pq_formulation(
            b=self,
            problem=self.pooling_problem,
            skip_product_quality=skip_product_quality,
        )
        self.inequalities = None

    def add_objective(self, use_flow_cost=False):
        assert self._built
        if use_flow_cost:
            return minimize_flow_cost_objective(b=self, problem=self.pooling_problem)
        else:
            return minimize_cost_objective(b=self, problem=self.pooling_problem)

    def add_inequalities(self, add_inequalities=True, add_uxt=True):
        del self.inequalities
        self.inequalities = pyo.Block()
        add_all_pooling_inequalities(
            self.inequalities, self, self.pooling_problem,
            add_variables=True,  # always add variables
            add_uxt=add_uxt,
            add_inequalities=add_inequalities
        )

    def add_cuts(self, add_inequalities=False):
        return add_valid_cuts(self.inequalities, self, self.pooling_problem, add_inequalities=add_inequalities)

    @property
    def flow_input_to_pool_to_output(self):
        return self.v

    @property
    def flow_pool_to_output(self):
        return self.y

    @property
    def flow_input_to_output(self):
        return self.z

    @property
    def fractional_flow_input_to_pool(self):
        return self.q
