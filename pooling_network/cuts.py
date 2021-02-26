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
from typing import List

import numpy as np
import pyomo.environ as pe
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.calculus.derivatives import differentiate

from pooling_network.network import Network
from pooling_network.pooling import (
    compute_beta_kl_bounds,
    compute_gamma_kl_bounds,
    problem_pool_output_qualities,
)
from pooling_network.inequalities import _generate_pooling_inequalities


def _gradient_cut_if_violated(block: pe.Block, expr, atol: float, diff_vars: List[_GeneralVarData]):
    expr_value = pe.value(expr)
    if True or not np.isclose(expr_value, 0.0, atol=atol) and expr_value > 0:
        diff_map = differentiate(expr, wrt_list=diff_vars)
        cut_expr = pe.value(expr) + sum(diff_map[i] * (v - pe.value(v)) for i, v in enumerate(diff_vars))
        return cut_expr <= 0
    return None


def _generate_valid_cuts(block: pe.Block, parent: pe.Block, pool_name: str, output_name: str, quality_name: str,
                         problem: Network, violation_threshold=1e-5):
    s = block.s[pool_name, output_name]
    y = block.y[output_name, quality_name, pool_name]
    t = block.t[output_name, quality_name, pool_name]
    var_cone = block.cut_var_cone[output_name, quality_name, pool_name]
    var_v = block.cut_var_v[output_name, quality_name, pool_name]

    gamma_lower, gamma_upper = compute_gamma_kl_bounds(
        pool_name, output_name, quality_name, problem
    )

    beta_lower, beta_upper = compute_beta_kl_bounds(
        pool_name, output_name, quality_name, problem
    )

    cut_info = {
        'pool': pool_name,
        'output': output_name,
        'quality': quality_name,
        'gamma_lower': gamma_lower,
        'gamma_upper': gamma_upper,
        'beta_lower': beta_lower,
        'beta_upper': beta_upper,
    }

    if gamma_lower is None or gamma_upper is None or beta_lower is None or beta_upper is None:
        return

    assert gamma_lower is not None and gamma_upper is not None
    assert beta_lower is not None and beta_upper is not None

    if beta_lower < 0:
        # Generate cut based on Equation 15
        var_v_value = pe.value(var_v, exception=False)
        s_value = pe.value(s, exception=False)
        var_cone_value = pe.value(var_cone, exception=False)
        if var_v_value is None or s_value is None or var_cone_value is None:
            return

        if np.isclose(s_value * var_cone_value, 0.0):
            return

        viol_cone = var_v_value - np.sqrt(s_value * var_cone_value)

        cut_info['type'] = 'cone'
        cut_info['viol'] = viol_cone

        if viol_cone > violation_threshold:
            # add cut for var_v^2 - s*var_cone <= 0
            prod_value = s_value * var_cone_value
            # Deal with numerical issues near the top of the cone
            if s_value > 0.001 or prod_value > 1e-6:
                s_sep_value = s_value
            else:
                s_sep_value = 0.001001

            if var_cone_value > 0.001 or prod_value > 1e-6:
                var_cone_sep_value = var_cone_value
            else:
                var_cone_sep_value = 0.001001

            # Recompute prod_val with new values
            prod_value = s_sep_value * var_cone_sep_value

            if prod_value > 1e-6:
                # Add cut!
                eq_value = var_v_value - np.sqrt(prod_value)
                deq_dvar_v = 1.0
                deq_ds = -0.5 * var_cone_value * (1/np.sqrt(prod_value))
                deq_dvar_cone = -0.5 * s_value * (1/np.sqrt(prod_value))

                expr = (
                    eq_value
                    + deq_dvar_v * (var_v - var_v_value)
                    + deq_ds * (s - s_value)
                    + deq_dvar_cone * (var_cone - var_cone_value)
                )

                yield expr <= 0, cut_info

    if beta_upper > 0 and gamma_lower < 0 and pe.value(y) > 0:
        # Generate cut based on Equation 18
        s_value = pe.value(s, exception=False)
        t_value = pe.value(t, exception=False)
        y_value = pe.value(y, exception=False)
        var_v_value = pe.value(var_v, exception=False)

        if s_value is None or t_value is None or y_value is None or var_v_value is None:
            return

        viol = (
            beta_upper * t_value
            + (gamma_upper - gamma_lower)*(beta_upper * s_value + y_value)
            - (beta_upper - gamma_lower)*var_v_value
            - gamma_lower * ( (var_v_value**2.0) / (y_value + var_v_value) )
            - beta_upper * gamma_upper
        )
        cut_info['type'] = 'nonlinear'
        cut_info['viol'] = viol
        if viol > violation_threshold:
            # Add cut!
            eq_value = (
                beta_upper*t_value
                + (gamma_upper - gamma_lower)*(beta_upper*s_value + y_value)
                - (beta_upper - gamma_lower)*var_v_value
                - gamma_lower * ( (var_v_value**2.0) / (y_value + var_v_value) )
            )
            deq_dt = beta_upper
            deq_ds = (gamma_upper - gamma_lower)*beta_upper
            deq_dy = (gamma_upper - gamma_lower) - (-gamma_upper*( (var_v_value**2.0) / (y_value + var_v_value)))
            deq_dvar_v = -(beta_upper - gamma_lower) - gamma_lower*(
                    2*var_v_value/(y_value + var_v_value)
                    - (var_v_value**2.0)/(y_value + var_v_value)
            )

            expr = (
                eq_value
                + deq_dt * (t - t_value)
                + deq_ds * (s - s_value)
                + deq_dy * (y - y_value)
                + deq_dvar_v * (var_v - var_v_value)
                - beta_upper*gamma_upper
            )

            yield expr <= 0, cut_info


def generate_valid_cuts(block: pe.Block, parent: pe.Block, problem: Network, violation_threshold=1e-5):
    for pool_name, output_name, quality_name in problem_pool_output_qualities(problem):
        yield from _generate_valid_cuts(block, parent, pool_name, output_name, quality_name, problem, violation_threshold)


def add_valid_cuts(block: pe.Block, parent: pe.Block, problem: Network, violation_threshold: float = 1e-5,
                   add_inequalities: bool = False):
    all_cuts_info = []
    for cut, cut_info in generate_valid_cuts(block, parent, problem, violation_threshold):
        block._cuts.add(cut)
        all_cuts_info.append(cut_info)

    if add_inequalities:
        for pool_name, output_name, quality_name in problem_pool_output_qualities(problem):
            for cut, cut_info in _generate_pooling_inequalities(block, parent, pool_name, output_name, quality_name, problem, violation_threshold=violation_threshold):
                block._cuts.add(cut)
                all_cuts_info.append(cut_info)

    return all_cuts_info
