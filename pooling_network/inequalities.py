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
import pyomo.environ as pe
from coramin.relaxations.mccormick import PWMcCormickRelaxation

from pooling_network.network import Network
from pooling_network.pooling import (
    compute_beta_kl_bounds,
    compute_gamma_ijk,
    compute_gamma_kl_bounds,
    problem_pool_output_qualities,
    index_set_lj,
    index_set_jkl
)


def _generate_pooling_inequalities(block: pe.Block, parent: pe.Block, pool_name: str, output_name: str,
                                   quality_name: str, problem: Network, violation_threshold: Optional[float] = None):
    gamma_lower, gamma_upper = compute_gamma_kl_bounds(
        pool_name, output_name, quality_name, problem
    )

    beta_lower, beta_upper = compute_beta_kl_bounds(
        pool_name, output_name, quality_name, problem
    )

    if beta_lower is None or beta_upper is None:
        return

    assert gamma_lower is not None and gamma_upper is not None
    assert beta_lower is not None and beta_upper is not None

    cut_info = {
        'pool': pool_name,
        'output': output_name,
        'quality': quality_name,
        'gamma_lower': gamma_lower,
        'gamma_upper': gamma_upper,
        'beta_lower': beta_lower,
        'beta_upper': beta_upper,
    }

    y = block.y[output_name, quality_name, pool_name]
    x = block.s[pool_name, output_name]
    u = block.u[output_name, quality_name, pool_name]
    t = block.t[output_name, quality_name, pool_name]

    if beta_lower < 0:
        # Eq 28
        expr = (
            (gamma_lower - beta_lower)*(gamma_upper*x - u)
            + beta_lower*(gamma_upper - t)
        )

        violated = True
        cut_info['type'] = 'inequality_28'
        cut_info['viol'] = 0.0

        if violation_threshold is not None:
            expr_value = pe.value(expr, exception=False)
            if expr_value is not None:
                violated = expr_value > violation_threshold
                cut_info['viol'] = expr_value

        if violated:
            yield expr <= 0, cut_info

    if beta_upper > 0:
        # Eq 22
        expr = (
            (gamma_upper - gamma_lower)*y
            + gamma_lower*(gamma_upper*x - u)
            + beta_upper*(u - gamma_lower * x)
            - beta_upper*(t - gamma_lower)
        )

        violated = True
        cut_info['type'] = 'inequality_22'
        cut_info['viol'] = 0.0

        if violation_threshold is not None:
            expr_value = pe.value(expr, exception=False)
            if expr_value is not None:
                violated = expr_value > violation_threshold
                cut_info['viol'] = expr_value

        if violated:
            yield expr <= 0, cut_info


def add_pooling_inequalities(block: pe.Block, parent: pe.Block, pool_name: str, output_name: str, quality_name: str,
                             problem: Network):
    for inequality, _ in _generate_pooling_inequalities(block, parent, pool_name, output_name, quality_name, problem):
        block._inequalities.add(inequality)


def _t_bounds(problem):
    def _bounds(m, j, k, l):
        return compute_gamma_kl_bounds(l, j, k, problem)
    return _bounds


def add_all_pooling_inequalities_variables(block: pe.Block, parent: pe.Block, problem: Network):
    block.z = pe.Var(index_set_lj(problem), bounds=(0, None))
    # s is the scaled parent.y[l, j]
    block.s = pe.Var(index_set_lj(problem), bounds=(0, None))
    block.u = pe.Var(index_set_jkl(problem))
    block.y = pe.Var(index_set_jkl(problem), initialize=1.0)
    block.t = pe.Var(index_set_jkl(problem), bounds=_t_bounds(problem))

    @block.Constraint(index_set_lj(problem))
    def z_def(b, l, j):
        output = problem.nodes[j]
        cap_lower, cap_upper = output.capacity
        return b.z[l, j] == (1/cap_upper) * sum(
            parent.z[input.name, j] for input in problem.predecessors(j, layer=0)
        ) + sum(
            parent.y[pool_.name, j] for pool_ in problem.predecessors(j, layer=1)
            if l != pool_.name
        )

    @block.Constraint(index_set_lj(problem))
    def s_def(b, l, j):
        output = problem.nodes[j]
        cap_lower, cap_upper = output.capacity
        return b.s[l, j] == (1 / cap_upper) * sum(
            parent.v[input.name, l, j]
            for input in problem.predecessors(l, layer=0)
        )

    @block.Constraint(index_set_jkl(problem))
    def u_def(b, j, k, l):
        output = problem.nodes[j]
        cap_lower, cap_upper = output.capacity
        return b.u[j, k, l] == (1/cap_upper) * sum(
            compute_gamma_ijk(input, output, k) * parent.v[input.name, l, j]
            for input in problem.predecessors(l, layer=0)
        )

    @block.Constraint(index_set_jkl(problem))
    def y_def(b, j, k, l):
        output = problem.nodes[j]
        cap_lower, cap_upper = output.capacity
        return b.y[j, k, l] == (1/cap_upper) * (
                sum(
                    compute_gamma_ijk(input, output, k) * parent.z[input.name, j]
                    for input in problem.predecessors(j, layer=0)
                ) + sum(
                    compute_gamma_ijk(input, output, k) * parent.v[input.name, pool_.name, j]
                    for pool_ in problem.predecessors(j, layer=1)
                    for input in problem.predecessors(pool_.name, layer=0)
                    if pool_.name != l
                )
        )

    @block.Constraint(index_set_jkl(problem))
    def t_def(b, j, k, l):
        # NOTE: this variable is not scaled.
        output = problem.nodes[j]
        return b.t[j, k, l] == sum(
            compute_gamma_ijk(i, output, k) * parent.q[i.name, l]
            for i in problem.predecessors(l, layer=0)
        )


def add_all_pooling_cuts_variables(block: pe.Block, parent: pe.Block, problem: Network):
    block.cut_var_v = pe.Var(index_set_jkl(problem))
    block.cut_var_cone = pe.Var(index_set_jkl(problem))

    @block.Constraint(index_set_jkl(problem))
    def cut_var_v_def(b, j, k, l):
        gamma_lower, _ = compute_gamma_kl_bounds(l, j, k, problem)
        return b.cut_var_v[j, k, l] == b.u[j, k, l] - gamma_lower*b.s[l, j]

    @block.Constraint(index_set_jkl(problem))
    def cut_var_cone_def(b, j, k, l):
        beta_lower, beta_upper = compute_beta_kl_bounds(l, j, k, problem)
        gamma_lower, gamma_upper = compute_gamma_kl_bounds(l, j, k, problem)
        if beta_lower is None or beta_lower >= 0:
            return pe.Constraint.Skip
        return b.cut_var_cone[j, k, l] == (
            - beta_lower*(b.t[j, k, l] - gamma_lower)
            + (beta_lower - gamma_lower)*(b.u[j, k, l] - gamma_lower*b.s[l, j])
        )


def add_all_ust_equations(block: pe.Block, parent: pe.Block, problem: Network):
    block.uxt = pe.Block()
    uxt_count = 1

    for pool_name, output_name, quality_name in problem_pool_output_qualities(problem):
        u = block.u[output_name, quality_name, pool_name]
        s = block.s[pool_name, output_name]
        t = block.t[output_name, quality_name, pool_name]
        rel = PWMcCormickRelaxation()
        rel.set_input(aux_var=u, x1=s, x2=t)
        setattr(block.uxt, 'rel' + str(uxt_count), rel)
        rel.rebuild()
        uxt_count += 1


def add_all_pooling_inequalities(block: pe.Block, parent: pe.Block, problem: Network, add_variables=True,
                                 add_inequalities=True, add_uxt=True):
    if add_variables:
        add_all_pooling_inequalities_variables(block, parent, problem)
        add_all_pooling_cuts_variables(block, parent, problem)

    block._inequalities = pe.ConstraintList()
    block._cuts = pe.ConstraintList()

    if add_inequalities:
        for pool_name, output_name, quality_name in problem_pool_output_qualities(problem):
            add_pooling_inequalities(
                block, parent, pool_name, output_name, quality_name, problem
            )

    if add_uxt:
        add_all_ust_equations(block, parent, problem)