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

from pooling_network.formulation.pq_block import PoolingPQFormulationData


def pooling_data_objects(block, descend_into=True, active=None, sort=False):
    for b in block.component_data_objects(pe.Block, descend_into=descend_into, active=active, sort=sort):
        if isinstance(b, PoolingPQFormulationData):
            yield b
