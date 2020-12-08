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
from pooling_network.network import Node, Network


def compute_gamma_ijk(input: Node, output: Node, quality_name: str):
    return input.attr['quality'][quality_name] - output.attr['quality_upper'][quality_name]


def compute_gamma_lower_ijk(input: Node, output: Node, quality_name: str):
    return input.attr['quality'][quality_name] - output.attr['quality_lower'][quality_name]


def compute_gamma_kl_bounds(pool_name: str, output_name: str, quality_name: str, problem: Network):
    min_quality = None
    max_quality = None
    output = problem.nodes[output_name]
    assert output.layer == 2
    for input in problem.predecessors(pool_name, layer=0):
        quality_diff = compute_gamma_ijk(input, output, quality_name)
        if min_quality is None:
            assert max_quality is None
            min_quality = max_quality = quality_diff
        else:
            min_quality = min(min_quality, quality_diff)
            max_quality = max(max_quality, quality_diff)
    return min_quality, max_quality


def compute_beta_kl_bounds(pool_name: str, output_name: str, quality_name: str, problem: Network):
    min_beta = None
    max_beta = None
    output = problem.nodes[output_name]
    for input in problem.predecessors(output_name, layer=0):
        quality_diff = compute_gamma_ijk(input, output, quality_name)
        if min_beta is None:
            assert max_beta is None
            min_beta = max_beta = quality_diff
        else:
            min_beta = min(min_beta, quality_diff)
            max_beta = max(max_beta, quality_diff)

    for bypass_pool in problem.predecessors(output_name, layer=1):
        if bypass_pool.name == pool_name:
            continue
        for input in problem.predecessors(bypass_pool.name, layer=0):
            quality_diff = compute_gamma_ijk(input, output, quality_name)
            if min_beta is None:
                assert max_beta is None
                min_beta = max_beta = quality_diff
            else:
                min_beta = min(min_beta, quality_diff)
                max_beta = max(max_beta, quality_diff)

    return min_beta, max_beta


def problem_pool_output_qualities(problem: Network):
    for pool in problem.nodes_at_layer(1):
        for output in problem.successors(pool.name, layer=2):
            for quality_name in output.attr['quality_upper'].keys():
                yield pool.name, output.name, quality_name


def index_set_i(network: Network):
    for node in network.nodes_at_layer(0):
        yield node.name


def index_set_l(network: Network):
    for node in network.nodes_at_layer(1):
        yield node.name


def index_set_j(network: Network):
    for node in network.nodes_at_layer(2):
        yield node.name


def index_set_il(network: Network):
    for input in network.nodes_at_layer(0):
        for pool in network.successors(input.name, layer=1):
            yield input.name, pool.name


def index_set_ilj(network: Network):
    for input in network.nodes_at_layer(0):
        for pool in network.successors(input.name, layer=1):
            for output in network.successors(pool.name, layer=2):
                yield input.name, pool.name, output.name


def index_set_lj(network: Network):
    for pool in network.nodes_at_layer(1):
        for output in network.successors(pool.name, layer=2):
            yield pool.name, output.name


def index_set_ij(network: Network):
    for input in network.nodes_at_layer(0):
        for output in network.successors(input.name, layer=2):
            yield input.name, output.name


def index_set_jk(network: Network):
    for output in network.nodes_at_layer(2):
        for k in output.attr['quality_upper'].keys():
            yield output.name, k


def index_set_jkl(network: Network):
    for output in network.nodes_at_layer(2):
        for pool in network.predecessors(output.name, layer=1):
            for k in output.attr['quality_upper'].keys():
                yield output.name, k, pool.name
