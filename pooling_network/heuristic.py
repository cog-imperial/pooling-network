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
from contextlib import contextmanager

import numpy as np
import pyomo.environ as pe

from pooling_network.network import Network
from pooling_network.pooling import (
    index_set_ilj,
    index_set_l
)


def uniform_restriction(tau):
    def func(t, l):
        return 1/tau
    return func


def _index_set_iltj(problem: Network, tau):
    for i in problem.nodes_at_layer(0):
        for l in problem.successors(i.name, layer=1):
            for t in range(tau):
                for j in problem.successors(l.name, layer=2):
                    yield i.name, l.name, t, j.name


def _index_set_ilt(problem: Network, tau):
    for i in problem.nodes_at_layer(0):
        for l in problem.successors(i.name, layer=1):
            for t in range(tau):
                yield i.name, l.name, t


def _index_set_ltj(problem: Network, tau):
    for l in problem.nodes_at_layer(1):
        for t in range(tau):
            for j in problem.successors(l.name, layer=2):
                yield l.name, t, j.name


def _index_set_lt(problem: Network, tau):
    for l in problem.nodes_at_layer(1):
        for t in range(tau):
            yield l.name, t


def add_mip_heuristic(parent: pe.Block, problem: Network, tau=2, restriction=None):
    """Build the MIP heuristic to `block` starting from the PQ-formulation in `parent`."""
    if restriction is None:
        restriction = uniform_restriction
    restriction_gen = restriction(tau)

    block = pe.Block()
    parent._pooling_mip_heuristic = block

    block.w = pe.Var(_index_set_iltj(problem, tau), bounds=(0, None))
    block.zeta = pe.Var(_index_set_ltj(problem, tau), domain=pe.Binary)

    @block.Constraint(index_set_ilj(problem))
    def flow_balance(b, i, l, j):
        return parent.v[i, l, j] == sum(
            block.w[i, l, t, j]
            for t in range(tau)
        )

    @block.Constraint(_index_set_ilt(problem, tau))
    def flow_balance_2(b, i, l, t):
        ws = [
            block.w[i, l, t, output.name]
            for output in problem.successors(l, layer=2)
        ]

        if not ws:
            return pe.Constraint.Skip

        return sum(ws) == restriction_gen(l, t) * sum(
            parent.v[i, l, output.name]
            for output in problem.successors(l, layer=2)
        )

    @block.Constraint(_index_set_iltj(problem, tau))
    def flow_choice_limit(b, i, l, t, j):
        edge = problem.edges[l, j]
        _, limit = edge.capacity
        return b.w[i, l, t, j] <= limit * b.zeta[l, t, j]

    @block.Constraint(_index_set_lt(problem, tau))
    def flow_choice(b, l, t):
        return sum(
            b.zeta[l, t, output.name] for output in problem.successors(l, layer=2)
        ) == 1

    # deactivate constraints
    parent.path_definition.deactivate()
    #parent.simplex.deactivate()
    parent.reduction_2.deactivate()


def remove_mip_heuristic(parent: pe.Block):
    # deactivate constraints
    parent.path_definition.activate()
    #parent.simplex.activate()
    parent.reduction_2.activate()
    del parent._pooling_mip_heuristic


@contextmanager
def mip_heuristic(parent: pe.Block, problem: Network, tau=2, restriction=None):
    add_mip_heuristic(parent, problem, tau, restriction)
    try:
        yield
    except Exception:
        remove_mip_heuristic(parent)
        raise
    remove_mip_heuristic(parent)


def derive_fractional_flow_variables(block):
    problem = block.pooling_problem

    # Set value of q_{il}
    for l in index_set_l(problem):
        x_il_value = dict()
        x_il_sum = 0.0
        num_inputs = 0
        for input in problem.predecessors(l, layer=0):
            i = input.name
            x_il = 0.0
            for output in problem.successors(l, layer=2):
                j = output.name
                v_ilj = pe.value(block.v[i, l, j], exception=False)
                if v_ilj is not None:
                    x_il += pe.value(block.v[i, l, j])
            x_il_value[i, l] = x_il
            x_il_sum += x_il
            num_inputs += 1

        for input in problem.predecessors(l, layer=0):
            i = input.name
            if np.isclose(x_il_sum, 0.0):
                value = 1 / num_inputs
            else:
                value = x_il_value[i, l] / x_il_sum
            block.q[i, l].set_value(value)
