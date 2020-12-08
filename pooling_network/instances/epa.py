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

import numpy as np
import pyomo.environ as pe

from pooling_network.network import Network
from pooling_network.pooling import index_set_j, index_set_jk, index_set_ilj, index_set_ij


def epa_case_1(region=1) -> Network:
    """
    References:
        Gounaris, C. E., & Floudas, C. A. (2007).
        Formulation and relaxation of an extended pooling problem.
        In 2007 AIChE Annual Meeting, Salt Lake City, Utah. AIChE.

        Misener, R., Gounaris, C. E., & Floudas, C. A. (2010).
        Mathematical modeling and global optimization of large-scale extended pooling problems with the (EPA)
        complex emissions constraints.
        Computers & Chemical Engineering, 34(9), 1432–1456.
    """
    # Data from Appendix B of Misener et al.
    qualities = ['OXY', 'SUL', 'RVP', 'E200', 'E300', 'ARO', 'BEN', 'OLE', 'MTB', 'ETB', 'ETH']

    # Table B.7
    num_feed = 7
    feed_cost = [2, 8, 10, 16, 2, 2, 5]
    feed_capacity_lower = [50, 0, 0, 0, 0, 0, 0]
    feed_capacity_upper = [400, 200, 200, 100, 10, 10, 50]
    # Table B.6
    feed_quality = \
        [[ 0.1,  800.0, 6.0,  20.0,  70.0, 50.0, 0.0, 10.0,  0.0,   0.0,   0.0],
         [ 0.2,  400.0, 8.8,  60.0,  85.0, 30.0, 0.8, 15.0,  0.0,   0.0,   0.0],
         [ 0.4,  200.0, 8.0,  55.0,  80.0, 25.0, 1.0, 15.0,  0.0,   0.0,   0.0],
         [ 0.7,  100.0, 8.0,  50.0,  75.0, 10.0, 0.2,  5.0,  0.0,   0.0,   0.0],
         [18.15,   0.0, 8.4, 100.0, 100.0,  0.0, 0.0,  0.0, 18.15,  0.0,   0.0],
         [15.66,   0.0, 8.0, 100.0, 100.0,  0.0, 0.0,  0.0,  0.0,  15.66,  0.0],
         [34.73,   0.0, 9.6, 100.0, 100.0,  0.0, 0.0,  0.0,  0.0,   0.0,  34.73]]

    assert len(feed_cost) == 7
    assert len(feed_capacity_lower) == 7
    assert len(feed_capacity_upper) == 7
    assert len(feed_quality) == 7
    assert all(len(f) == 11 for f in feed_quality)

    # Table B.8
    num_product = 2
    product_cost = [6, 12]
    product_capacity_lower = [100, 100]
    product_capacity_upper = [200, 200]
    # Table B.6
    product_quality_lower = \
        [[0.3, 50, 6.4, 30, 70, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
         [0.3, 50, 6.4, 30, 70, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1]]
    product_quality_upper = \
        [[4.0, 500, 10.0, 70, 100, 30, 2.0, 25, 4.0, 4.0, 4.0],
         [4.0, 250,  8.0, 60,  85, 25, 0.5, 10, 4.0, 4.0, 4.0]]

    product_vb = [-2.77929,  -2.26558]
    product_nb = [ 0.497032, -0.179906]
    product_bb = [ 1.26152,   1.76845]
    product_fb = [-1.07807,  -1.36651]
    product_ab = [-0.751747, -1.09751]
    product_db = [-1.34704,  -0.694224]
    product_wv = [ 0.444,     0.556]
    product_wn = [ 0.738,     0.262]
    product_wt = [ 0.444,     0.556]

    product_cv = \
        [[-0.003641,  0.0005219, 0.0289749, -0.014470, -0.068624, 0.0323712, -0.002858, 0.0001072, 0.0004087, -0.0003481],
         [-0.003626, -0.000054,  0.043295,  -0.013504, -0.062327, 0.0282042, -0.002858, 0.000106,  0.000408,  -0.000287]]

    product_cn = \
        [[ 0.00178571, 0.0006921, 0.0090744, 0.000931,  0.000846, 0.0083632, -0.002774, -0.000000663, -0.000119,   0.0003665],
         [-0.00913,    0.000252, -0.01397,   0.000931, -0.00401,  0.007097,  -0.00276,   0.0,         -0.00007995, 0.0003665]]

    product_cbe = \
        [[ 0.0,      0.0006197, -0.003376, 0.02655,  0.22239],
         [-0.096047, 0.000337,   0.011251, 0.011882, 0.222318]]

    product_cf = \
        [[-0.010226, -0.007166,  0.0,      0.0462131],
         [-0.010226, -0.007166, -0.031352, 0.0462131]]

    product_ca = \
        [[0.0002631, 0.039786, -0.012172, -0.005525, -0.009594, 0.31658,   0.2492500],
         [0.0002627, 0.0,      -0.012157, -0.005548, -0.05598,  0.3164665, 0.2493259]]

    product_cbu = \
        [[ 0.0,      0.0001552, -0.007253, -0.014866, -0.004005, 0.028235],
         [-0.060771, 0.0,       -0.007311, -0.008052, -0.004005, 0.043696]]

    if region == 1:
        quality_alpha_v = [1.2269, -0.3534, 0.0318]
    else:
        quality_alpha_v = [1.0633, -0.3008, 0.0270]

    if region == 1:
        quality_alpha_nb = \
            [1.75021, -0.603184, -0.0402619, 0.0738116, 0.0116427, -0.00255327, -0.0010494]
    else:
        quality_alpha_nb = \
            [1.5210, -0.5161, -0.0352, 0.0628, 0.0100, -0.0022, -0.0009]

    assert len(product_cost) == 2
    assert len(product_capacity_lower) == 2
    assert len(product_capacity_upper) == 2
    assert len(product_quality_lower) == 2
    assert len(product_quality_upper) == 2
    assert all(len(q) == 11 for q in product_quality_upper)
    assert all(len(q) == 11 for q in product_quality_lower)

    net = Network('epa_case_1')

    for i in range(num_feed):
        quality = dict([(qualities[qi], q) for qi, q in enumerate(feed_quality[i])])

        net.add_node(
            layer=0,
            name='Feed{}'.format(i),
            capacity_lower=feed_capacity_lower[i],
            capacity_upper=feed_capacity_upper[i],
            cost=feed_cost[i],
            attr={
                'quality': quality,
            }
        )

    net.add_node(
        layer=1,
        name='Pool0',
        capacity_lower=0.0,
        capacity_upper=300.00,
        cost=0.0,
        attr=dict(),
    )

    for j in range(num_product):
        quality_lower = dict([(qualities[qi], q) for qi, q in enumerate(product_quality_lower[j])])
        quality_upper = dict([(qualities[qi], q) for qi, q in enumerate(product_quality_upper[j])])

        net.add_node(
            layer=2,
            name='Product{}'.format(j),
            capacity_lower=product_capacity_lower[j],
            capacity_upper=product_capacity_upper[j],
            cost=product_cost[j],
            attr={
                'quality_lower': quality_lower,
                'quality_upper': quality_upper,
                'vb': product_vb[j],
                'nb': product_nb[j],
                'bb': product_bb[j],
                'fb': product_fb[j],
                'ab': product_ab[j],
                'db': product_db[j],
                'wv': product_wv[j],
                'wn': product_wn[j],
                'wt': product_wt[j],
                'cv': product_cv[j],
                'cn': product_cn[j],
                'cbe': product_cbe[j],
                'cf': product_cf[j],
                'ca': product_ca[j],
                'cbu': product_cbu[j],
            }
        )

    for i in range(4):
        net.add_edge('Feed{}'.format(i), 'Pool0', cost=0.0, fixed_cost=0.0)

    for j in range(2):
        net.add_edge(
            'Pool0',
            'Product{}'.format(j),
            cost=0.0,
            fixed_cost=0.0
        )

        for i in range(4):
            net.add_edge(
                'Feed{}'.format(i+3),
                'Product{}'.format(j),
                cost=0.0,
                fixed_cost=0.0,
            )

    net.attr = {
        'voc_max': 1200,
        'nox_max': 1300,
        'tox_max': 90,
        'vocne_coeff': 1000,
        'nebenz_coeff': 10,
        'voce_b': 907,
        'nox_b': 1340,
        'benz_b': 53.54,
        'form_b': 9.7,
        'acet_b': 4.44,
        'buta_b': 9.38,
        'alpha_v': quality_alpha_v,
        'alpha_nb': quality_alpha_nb,
    }

    return net


def _input_quality(problem, i, k):
    return problem.nodes[i].attr['quality'][k]


def _index_set_je(network: Network):
    for output_0 in network.nodes_at_layer(2):
        for output_1 in network.nodes_at_layer(2):
            yield output_0.name, output_1.name


def add_epa_constraints(block, pooling_block):
    problem = pooling_block.pooling_problem

    block.of = pe.Var(index_set_j(problem), bounds=(_of_bounds(problem)))
    block.u = pe.Var(index_set_jk(problem), bounds=(_u_bounds(problem)))

    _add_flow_constraints(block, pooling_block, problem)
    _add_logical_disjunctions(block, pooling_block, problem)
    _add_emission_models(block, pooling_block, problem)


def _of_bounds(problem: Network):
    def _bounds(m, j):
        output = problem.nodes[j]
        return output.capacity_lower, output.capacity_upper
    return _bounds


def _u_bounds(problem: Network):
    def _bounds(m, j, k):
        output = problem.nodes[j]
        quality_lower = output.attr.get('quality_lower')
        quality_upper = output.attr.get('quality_upper')
        assert quality_lower is not None
        assert quality_upper is not None
        lb = max(
            quality_lower[k],
            min(input.attr['quality'][k] for input in problem.nodes_at_layer(layer=0))
        )
        ub = min(
            quality_upper[k],
            max(input.attr['quality'][k] for input in problem.nodes_at_layer(layer=0))
        )
        # RVP blends non linearly
        if k == 'RVP':
            lb = quality_lower[k]
            ub = quality_upper[k]
        return lb, ub
    return _bounds


def _add_flow_constraints(block, pooling_block, problem):
    block.flow_constraints = pe.Block()
    b = block.flow_constraints

    @b.Constraint(index_set_j(problem))
    def of_def(m, j):
        return block.of[j] == sum(
            pooling_block.y[l.name, j] for l in problem.predecessors(j, layer=1)
        ) + sum(
            pooling_block.z[i.name, j] for i in problem.predecessors(j, layer=0)
        )

    @b.Constraint(index_set_jk(problem))
    def of_balance(m, j, k):
        if k == 'RVP':
            return (block.u[j, k]**1.25)*block.of[j] == sum(
                (_input_quality(problem, i, k)**1.25) * pooling_block.v[i, l, j]
                for i, l, j_ in index_set_ilj(problem) if j_ == j
            ) + sum(
                (_input_quality(problem, i, k)**1.25) * pooling_block.z[i, j]
                for i, j_ in index_set_ij(problem) if j_ == j
            )
        else:
            return block.u[j, k] * block.of[j] == sum(
                _input_quality(problem, i, k) * pooling_block.v[i, l, j]
                for i, l, j_ in index_set_ilj(problem) if j_ == j
            ) + sum(
                _input_quality(problem, i, k) * pooling_block.z[i, j]
                for i, j_ in index_set_ij(problem) if j_ == j
            )

    @b.Constraint(index_set_j(problem))
    def oxygen_balance(m, j):
        return block.u[j, 'OXY'] >= block.u[j, 'MTB'] + block.u[j, 'ETB'] + block.u[j, 'ETH']


def _add_logical_disjunctions(block, pooling_block, problem):
    block.logical_disjunctions = pe.Block()
    b = block.logical_disjunctions

    b.e300_95 = _logical_disjunction_block(block, problem, 'E300', 95)
    b.aro_10 = _logical_disjunction_block(block, problem, 'ARO', 10)

    b.v_e200_33 = _logical_disjunction_block(block, problem, 'E200', 33)
    b.v_e200_6552 = _logical_disjunction_block(block, problem, 'E200', 65.52)
    b.v_e300_72 = _logical_disjunction_block(block, problem, 'E300', 72)
    b.v_aro_18 = _logical_disjunction_block(block, problem, 'ARO', 18)
    b.v_aro_46 = _logical_disjunction_block(block, problem, 'ARO', 46)

    b.n_sul_10 = _logical_disjunction_block(block, problem, 'SUL', 10)
    b.n_sul_450 = _logical_disjunction_block(block, problem, 'SUL', 450)
    b.n_aro_368 = _logical_disjunction_block(block, problem, 'ARO', 36.8)
    b.n_ole_377 = _logical_disjunction_block(block, problem, 'OLE', 3.77)
    b.n_ole_19 = _logical_disjunction_block(block, problem, 'OLE', 19)

    b.n_aro_18 = pe.Block()
    b.n_aro_18.y = pe.Var(index_set_j(problem), domain=pe.Binary)

    @b.n_aro_18.Constraint(index_set_j(problem))
    def eq1(m, j):
        return m.y[j] == b.v_aro_18.y[j]

    b.v_star = pe.Block()
    b.v_star.y = pe.Var(index_set_j(problem), domain=pe.Binary)

    @b.v_star.Constraint(index_set_j(problem))
    def eq1(m, j):
        diff = 0.385*block.u[j, 'ARO'].ub - 14.25
        lhs = 0.385*block.u[j, 'ARO'] - diff * m.y[j]
        return lhs <= 14.25

    @b.v_star.Constraint(index_set_j(problem))
    def eq2(m, j):
        diff = 0.385*block.u[j, 'ARO'].lb - 14.25
        lhs = 0.385*block.u[j, 'ARO'] + diff * m.y[j]
        return lhs >= 14.25 + diff

    b.v_star_e300 = pe.Block()
    b.v_star_e300.y = pe.Var(index_set_j(problem), domain=pe.Binary)


def _logical_disjunction_block(parent_block, problem, quality, bound):
    block = pe.Block()
    block.y = pe.Var(index_set_j(problem), domain=pe.Binary)

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        lhs = (parent_block.u[j, quality].lb - bound)*m.y[j] - parent_block.u[j, quality]
        return lhs <= -bound

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        diff = parent_block.u[j, quality].ub - bound
        lhs = parent_block.u[j, quality] + diff*m.y[j]
        rhs = diff + bound
        return lhs <= rhs

    return block


def _add_emission_models(block, pooling_problem, problem):
    block.emission_models = pe.Block()
    b = block.emission_models

    def sul_f_eq1_rhs(m, j, e):
        cn = problem.nodes[e].attr['cn']
        c = 2*cn[7]*10 + cn[1]
        return c * block.u[j, 'SUL'] - c*10

    def sul_f_eq2_rhs(m, j, e):
        return 0

    def sul_f_eq3_rhs(m, j, e):
        cn = problem.nodes[e].attr['cn']
        c = 2*cn[7]*450 + cn[1]
        return c * block.u[j, 'SUL'] - c*450

    b.sul = _emission_model_block_1(
        block,
        problem,
        _f_sul_bounds,
        block.logical_disjunctions.n_sul_10,
        block.logical_disjunctions.n_sul_450,
        sul_f_eq1_rhs,
        sul_f_eq2_rhs,
        sul_f_eq3_rhs,
    )

    def aro_f_eq1_rhs(m, j, e):
        cn = problem.nodes[e].attr['cn']
        c = 2*cn[8]*18 + cn[5]
        return -8*c

    def aro_f_eq2_rhs(m, j, e):
        cn = problem.nodes[e].attr['cn']
        c = 2*cn[8]*18 + cn[5]
        return c*block.u[j, 'ARO'] - 18*c

    def aro_f_eq3_rhs(m, j, e):
        return 0.0

    b.aro = _emission_model_block_1(
        block,
        problem,
        _f_aro_bounds,
        block.logical_disjunctions.aro_10,
        block.logical_disjunctions.n_aro_18,
        aro_f_eq1_rhs,
        aro_f_eq2_rhs,
        aro_f_eq3_rhs,
    )

    def ole_f_eq1_rhs(m, j, e):
        return 0.0

    def ole_f_eq2_rhs(m, j, e):
        cn = problem.nodes[e].attr['cn']
        c = 2*cn[9]*19 + cn[6]
        return c*block.u[j, 'OLE'] - 19*c

    b.ole = _emission_model_block_2(
        block,
        problem,
        _f_ole_bounds,
        block.logical_disjunctions.n_ole_19,
        ole_f_eq1_rhs,
        ole_f_eq2_rhs,
    )

    def e200_f_eq1_rhs(m, j, e):
        cv = problem.nodes[e].attr['cv']
        c = 2*cv[7]*33 + cv[3]
        return c*block.u[j, 'E200'] - 33*c

    def e200_f_eq2_rhs(m, j, e):
        return 0.0

    b.e200 = _emission_model_block_2(
        block,
        problem,
        _f_e200_bounds,
        block.logical_disjunctions.v_e200_33,
        e200_f_eq1_rhs,
        e200_f_eq2_rhs,
    )

    b.rvp = pe.Var(index_set_j(problem), bounds=rvp_bounds(block, problem))

    b.e200_et = _e200_et_block(block, problem, block.logical_disjunctions)
    b.aro_et = _aro_et_block(block, problem, block.logical_disjunctions)
    b.aro_del = _aro_del_block(block, problem, block.logical_disjunctions)
    b.e300_star = _e300_star_block(block, problem, block.logical_disjunctions)
    b.e300_extension = _e300_extension_block(block, problem, block.logical_disjunctions)
    b.e300_del_extension = _e300_del_extension_block(block, problem, block.logical_disjunctions)
    b.sul_et_nox = _sul_et_nox_block(block, problem, block.logical_disjunctions)
    b.e300_et_nox = _e300_et_nox_block(block, problem, block.logical_disjunctions)
    b.aro_et_nox = _aro_et_nox_block(block, problem, block.logical_disjunctions)
    b.ole_et_nox = _ole_et_nox_block(block, problem, block.logical_disjunctions)
    b.e300_et_tox = _e300_et_tox_block(block, problem, block.logical_disjunctions)
    b.aro_et_tox = _aro_et_tox_block(block, problem, block.logical_disjunctions)

    b.voc = _voc_block(block, problem, block.logical_disjunctions)
    b.nox = _nox_block(block, problem, block.logical_disjunctions)
    b.tox = _tox_block(block, problem, block.logical_disjunctions)


def _emission_model_block_1(parent_block, problem, f_bounds, disj_block, disj2_block,
                            f_eq1_rhs, f_eq2_rhs, f_eq3_rhs):

    block = pe.Block()

    _add_emission_model_block_common(block, parent_block, problem, f_bounds)

    @block.Constraint(_index_set_je(problem))
    def s_eq1(m, j, e):
        lhs = m.lo.p[j, e] + m.lo.m[j, e] + m.u[j, e]*disj_block.y[j]
        return lhs <= m.u[j, e]

    @block.Constraint(_index_set_je(problem))
    def s_eq2(m, j, e):
        lhs = m.md.p[j, e] + m.md.m[j, e] + m.u[j, e]*disj2_block.y[j] - m.u[j, e]*disj_block.y[j]
        return lhs <= m.u[j, e]

    @block.Constraint(_index_set_je(problem))
    def s_eq3(m, j, e):
        lhs = m.up.p[j, e] + m.up.m[j, e] - m.u[j, e]*disj2_block.y[j]
        return lhs <= 0.0

    @block.Constraint(_index_set_je(problem))
    def f_eq1(m, j, e):
        lhs = m.f[j, e] + m.lo.p[j, e] - m.lo.m[j, e]
        rhs = f_eq1_rhs(m, j, e)
        return lhs == rhs

    @block.Constraint(_index_set_je(problem))
    def f_eq2(m, j, e):
        lhs = m.f[j, e] + m.md.p[j, e] - m.md.m[j, e]
        rhs = f_eq2_rhs(m, j, e)
        return lhs == rhs

    @block.Constraint(_index_set_je(problem))
    def f_eq3(m, j, e):
        lhs = m.f[j, e] + m.up.p[j, e] - m.up.m[j, e]
        rhs = f_eq3_rhs(m, j, e)
        return lhs == rhs

    return block


def _emission_model_block_2(parent_block, problem, f_bounds, disj_block, f_eq1_rhs, f_eq2_rhs):
    block = pe.Block()

    _add_emission_model_block_common(block, parent_block, problem, f_bounds, skip_md=True)

    @block.Constraint(_index_set_je(problem))
    def s_eq1(m, j, e):
        lhs = m.lo.p[j, e] + m.lo.m[j, e] + m.u[j, e]*disj_block.y[j]
        return lhs <= m.u[j, e]

    @block.Constraint(_index_set_je(problem))
    def s_eq2(m, j, e):
        lhs = m.up.p[j, e] + m.up.m[j, e] - m.u[j, e]*disj_block.y[j]
        return lhs <= 0.0

    @block.Constraint(_index_set_je(problem))
    def f_eq1(m, j, e):
        lhs = m.f[j, e] + m.lo.p[j, e] - m.lo.m[j, e]
        rhs = f_eq1_rhs(m, j, e)
        return lhs == rhs

    @block.Constraint(_index_set_je(problem))
    def f_eq2(m, j, e):
        lhs = m.f[j, e] + m.up.p[j, e] - m.up.m[j, e]
        rhs = f_eq2_rhs(m, j, e)
        return lhs == rhs

    return block


def _add_emission_model_block_common(block, parent_block, problem, f_bounds, skip_md=False):
    block.f = pe.Var(_index_set_je(problem), bounds=f_bounds(parent_block, problem))

    block.u = pe.Param(
        _index_set_je(problem),
        initialize=lambda m, j, e: m.f[j, e].ub - m.f[j, e].lb
    )

    block.up = _emission_model_range_block(block, problem)
    if not skip_md:
        block.md = _emission_model_range_block(block, problem)
    block.lo = _emission_model_range_block(block, problem)


def _emission_model_range_block(parent_block, problem):
    block = pe.Block()
    block.p = pe.Var(
        _index_set_je(problem),
        bounds=lambda m, j, e: (0, parent_block.u[j, e])
    )
    block.m = pe.Var(
        _index_set_je(problem),
        bounds=lambda m, j, e: (0, parent_block.u[j, e])
    )
    return block


def _e200_et_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_e200_voc_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        return m.voc[j] - 33 <= (m.voc[j].ub - 33)*(1 - disj_block.v_e200_33.y[j])

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        return m.voc[j] - 33 >= (m.voc[j].lb - 33)*(1 - disj_block.v_e200_33.y[j])

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (m.voc[j].ub - parent_block.u[j, 'E200'].lb)*(1 - disj_block.v_e200_6552.y[j] + disj_block.v_e200_33.y[j])
        return m.voc[j] - parent_block.u[j, 'E200'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (m.voc[j].lb - parent_block.u[j, 'E200'].ub)*(1 - disj_block.v_e200_6552.y[j] + disj_block.v_e200_33.y[j])
        return m.voc[j] - parent_block.u[j, 'E200'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        return m.voc[j] - 65.52 <= (m.voc[j].ub - 65.52)*disj_block.v_e200_6552.y[j]

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        return m.voc[j] - 65.52 >= (m.voc[j].lb - 65.52)*disj_block.v_e200_6552.y[j]

    return block


def _aro_et_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_aro_voc_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        return m.voc[j] - 18 <= (parent_block.u[j, 'ARO'].ub - 18)*(1 - disj_block.v_aro_18.y[j])

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        return m.voc[j] - 18 >= (parent_block.u[j, 'ARO'].lb - 18)*(1 - disj_block.v_aro_18.y[j])

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - parent_block.u[j, 'ARO'].lb)*(1 - disj_block.v_aro_46.y[j] + disj_block.v_aro_18.y[j])
        return m.voc[j] - parent_block.u[j, 'ARO'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - parent_block.u[j, 'ARO'].ub)*(1 - disj_block.v_aro_46.y[j] + disj_block.v_aro_18.y[j])
        return m.voc[j] - parent_block.u[j, 'ARO'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        return m.voc[j] - 46.0 <= (parent_block.u[j, 'ARO'].ub - 46.0)*disj_block.v_aro_46.y[j]

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        return m.voc[j] - 46.0 >= (parent_block.u[j, 'ARO'].lb - 46.0)*disj_block.v_aro_46.y[j]

    return block


def _aro_del_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_aro_voc_del_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        return m.voc[j] + 8 <= (m.voc[j].ub + 8)*(1 - disj_block.aro_10.y[j])

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        return m.voc[j] + 8 >= (m.voc[j].lb + 8)*(1 - disj_block.aro_10.y[j])

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (m.voc[j].ub - parent_block.u[j, 'ARO'].lb + 18)*(disj_block.aro_10.y[j] + 1 - disj_block.v_aro_18.y[j])
        return m.voc[j] - parent_block.u[j, 'ARO'] + 18 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (m.voc[j].lb - parent_block.u[j, 'ARO'].ub + 18)*(disj_block.aro_10.y[j] + 1 - disj_block.v_aro_18.y[j])
        return m.voc[j] - parent_block.u[j, 'ARO'] + 18 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        return m.voc[j] <= m.voc[j].ub*(disj_block.v_aro_18.y[j] + 1 - disj_block.v_aro_46.y[j])

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        return m.voc[j] >= m.voc[j].lb*(disj_block.v_aro_18.y[j] + 1 - disj_block.v_aro_46.y[j])

    @block.Constraint(index_set_j(problem))
    def eq7(m, j):
        rhs = (m.voc[j].ub - parent_block.u[j, 'ARO'].lb + 46)*disj_block.v_aro_46.y[j]
        return m.voc[j] - parent_block.u[j, 'ARO'] + 46 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq8(m, j):
        rhs = (m.voc[j].lb - parent_block.u[j, 'ARO'].ub + 46)*disj_block.v_aro_46.y[j]
        return m.voc[j] - parent_block.u[j, 'ARO'] + 46 >= rhs

    return block


def _e300_star_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_e300_voc_star_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = 0.385*(parent_block.u[j, 'ARO'].ub - 10)*(disj_block.v_star.y[j] + 1 - disj_block.aro_10.y[j])
        return m.voc[j] - 0.385*10 - 79.75 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = 0.385*(parent_block.u[j, 'ARO'].lb - 10)*(disj_block.v_star.y[j] + 1 - disj_block.aro_10.y[j])
        return m.voc[j] - 0.385*10 - 79.75 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = 0.385*(parent_block.u[j, 'ARO'].ub - parent_block.u[j, 'ARO'].lb)*(disj_block.v_star.y[j] + disj_block.aro_10.y[j])
        return m.voc[j] - 0.385*parent_block.u[j, 'ARO'] - 79.75 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = 0.385*(parent_block.u[j, 'ARO'].lb - parent_block.u[j, 'ARO'].ub)*(disj_block.v_star.y[j] + disj_block.aro_10.y[j])
        return m.voc[j] - 0.385*parent_block.u[j, 'ARO'] - 79.75 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = (0.385*parent_block.u[j, 'ARO'].ub + 79.75 - 94.0)*(1 - disj_block.v_star.y[j])
        return m.voc[j] - 94 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = (0.385*parent_block.u[j, 'ARO'].lb + 79.75 - 94.0)*(1 - disj_block.v_star.y[j])
        return m.voc[j] - 94 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq7(m, j):
        rhs = (parent_block.u[j, 'E300'].lb - 0.385*parent_block.u[j, 'ARO'].ub - 79.75)*disj_block.v_star_e300.y[j]
        return parent_block.u[j, 'E300'] - m.voc[j] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq8(m, j):
        rhs = (parent_block.u[j, 'E300'].ub - 0.385*10.0 - 79.75)*(1 - disj_block.v_star_e300.y[j])
        return parent_block.u[j, 'E300'] - m.voc[j] <= rhs

    return block


def _e300_extension_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_e300_voc_bounds(parent_block))

    # EdGe TaRgEt
    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        return m.voc[j] - 72 <= (parent_block.u[j, 'E300'].ub - 72)*(1 - disj_block.v_e300_72.y[j])

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        return m.voc[j] - 72 >= (parent_block.u[j, 'E300'].lb - 72)*(1 - disj_block.v_e300_72.y[j])

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'E300'].ub - parent_block.u[j, 'E300'].lb)*(disj_block.v_e300_72.y[j] + 1 - disj_block.v_star_e300.y[j])
        return m.voc[j] - parent_block.u[j, 'E300'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'E300'].lb - parent_block.u[j, 'E300'].ub)*(disj_block.v_e300_72.y[j] + 1 - disj_block.v_star_e300.y[j])
        return m.voc[j] - parent_block.u[j, 'E300'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = (parent_block.u[j, 'E300'].ub - 0.385*10 - 79.75)*disj_block.v_star_e300.y[j]
        return m.voc[j] - parent_block.emission_models.e300_star.voc[j] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = (parent_block.u[j, 'E300'].lb - 0.385*parent_block.u[j, 'ARO'].ub - 79.75)*disj_block.v_star_e300.y[j]
        return m.voc[j] - parent_block.emission_models.e300_star.voc[j] >= rhs

    return block


def _e300_del_extension_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voc = pe.Var(index_set_j(problem), bounds=_e300_voc_del_bounds(parent_block))

    # DeLtA eXtEnSiOn
    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (1 - disj_block.v_e300_72.y[j])*(m.voc[j].ub - parent_block.u[j, 'E300'].lb + 72)
        return m.voc[j] - parent_block.u[j, 'E300'] + 72 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (1 - disj_block.v_e300_72.y[j])*(m.voc[j].lb - parent_block.u[j, 'E300'].ub + 72)
        return m.voc[j] - parent_block.u[j, 'E300'] + 72 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = m.voc[j].lb*(1 - disj_block.v_star_e300.y[j] + disj_block.v_e300_72.y[j])
        return m.voc[j] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = m.voc[j].ub*(1 - disj_block.v_star_e300.y[j] + disj_block.v_e300_72.y[j])
        return m.voc[j] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = m.voc[j].lb*(disj_block.v_star.y[j] + disj_block.v_e300_72.y[j])
        return m.voc[j] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = m.voc[j].ub*(disj_block.v_star.y[j] + disj_block.v_e300_72.y[j])
        return m.voc[j] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq7(m, j):
        rhs = (disj_block.v_star_e300.y[j] + 1 - disj_block.v_star.y[j])*(m.voc[j].lb - 95 + 94)
        return m.voc[j] - parent_block.u[j, 'E300'] + 94 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq8(m, j):
        rhs = (disj_block.v_star_e300.y[j] + 1 - disj_block.v_star.y[j])*(m.voc[j].ub - parent_block.emission_models.e300_extension.voc[j].lb + 94)
        return m.voc[j] - parent_block.u[j, 'E300'] + 94 <= rhs

    return block


def _sul_et_nox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.nox = pe.Var(index_set_j(problem), bounds=_sul_nox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (parent_block.u[j, 'SUL'].ub - 10)*(1 - disj_block.n_sul_10.y[j])
        return m.nox[j] - 10 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (parent_block.u[j, 'SUL'].lb - 10)*(1 - disj_block.n_sul_10.y[j])
        return m.nox[j] - 10 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'SUL'].ub - parent_block.u[j, 'SUL'].lb)*(1 - disj_block.n_sul_450.y[j] + disj_block.n_sul_10.y[j])
        return m.nox[j] - parent_block.u[j, 'SUL'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'SUL'].lb - parent_block.u[j, 'SUL'].ub)*(1 - disj_block.n_sul_450.y[j] + disj_block.n_sul_10.y[j])
        return m.nox[j] - parent_block.u[j, 'SUL'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = (parent_block.u[j, 'SUL'].ub - 450)*disj_block.n_sul_450.y[j]
        return m.nox[j] - 450 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = (parent_block.u[j, 'SUL'].lb - 450)*disj_block.n_sul_450.y[j]
        return m.nox[j] - 450 >= rhs

    return block


def _e300_et_nox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.nox = pe.Var(index_set_j(problem), bounds=_e300_nox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (parent_block.u[j, 'E300'].ub - 95)*(disj_block.e300_95.y[j])
        return m.nox[j] - 95.0 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (parent_block.u[j, 'E300'].lb - 95)*(disj_block.e300_95.y[j])
        return m.nox[j] - 95.0 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'E300'].ub - parent_block.u[j, 'E300'].lb)*(1 - disj_block.e300_95.y[j])
        return m.nox[j] - parent_block.u[j, 'E300'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'E300'].lb - parent_block.u[j, 'E300'].ub)*(1 - disj_block.e300_95.y[j])
        return m.nox[j] - parent_block.u[j, 'E300'] >= rhs

    return block


def _aro_et_nox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.nox = pe.Var(index_set_j(problem), bounds=_aro_nox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - 18)*(1 - disj_block.n_aro_18.y[j])
        return m.nox[j] - 18 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - 18)*(1 - disj_block.n_aro_18.y[j])
        return m.nox[j] - 18 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - parent_block.u[j, 'ARO'].lb)*(1 - disj_block.n_aro_368.y[j] + disj_block.n_aro_18.y[j])
        return m.nox[j] - parent_block.u[j, 'ARO'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - parent_block.u[j, 'ARO'].ub)*(1 - disj_block.n_aro_368.y[j] + disj_block.n_aro_18.y[j])
        return m.nox[j] - parent_block.u[j, 'ARO'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - 36.8)*disj_block.n_aro_368.y[j]
        return m.nox[j] - 36.8 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - 36.8)*disj_block.n_aro_368.y[j]
        return m.nox[j] - 36.8 >= rhs

    return block


def _ole_et_nox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.nox = pe.Var(index_set_j(problem), bounds=_ole_nox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (parent_block.u[j, 'OLE'].ub - 3.77)*(1 - disj_block.n_ole_377.y[j])
        return m.nox[j] - 3.77 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (parent_block.u[j, 'OLE'].lb - 3.77)*(1 - disj_block.n_ole_377.y[j])
        return m.nox[j] - 3.77 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'OLE'].ub - parent_block.u[j, 'OLE'].lb)*(1 - disj_block.n_ole_19.y[j] + disj_block.n_ole_377.y[j])
        return m.nox[j] - parent_block.u[j, 'OLE'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'OLE'].lb - parent_block.u[j, 'OLE'].ub)*(1 - disj_block.n_ole_19.y[j] + disj_block.n_ole_377.y[j])
        return m.nox[j] - parent_block.u[j, 'OLE'] >= rhs

    @block.Constraint(index_set_j(problem))
    def eq5(m, j):
        rhs = (parent_block.u[j, 'OLE'].ub - 19)*disj_block.n_ole_19.y[j]
        return m.nox[j] - 19 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq6(m, j):
        rhs = (parent_block.u[j, 'OLE'].lb - 19)*disj_block.n_ole_19.y[j]
        return m.nox[j] - 19 >= rhs

    return block


def _e300_et_tox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.tox = pe.Var(index_set_j(problem), bounds=_e300_tox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        return parent_block.emission_models.e300_et_nox.nox[j] == m.tox[j]

    return block


def _aro_et_tox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.tox = pe.Var(index_set_j(problem), bounds=_aro_tox_bounds(parent_block))

    @block.Constraint(index_set_j(problem))
    def eq1(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - 10)*(1 - disj_block.aro_10.y[j])
        return m.tox[j] - 10 <= rhs

    @block.Constraint(index_set_j(problem))
    def eq2(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - 10)*(1 - disj_block.aro_10.y[j])
        return m.tox[j] - 10 >= rhs

    @block.Constraint(index_set_j(problem))
    def eq3(m, j):
        rhs = (parent_block.u[j, 'ARO'].ub - parent_block.u[j, 'ARO'].lb)*disj_block.aro_10.y[j]
        return m.tox[j] - parent_block.u[j, 'ARO'] <= rhs

    @block.Constraint(index_set_j(problem))
    def eq4(m, j):
        rhs = (parent_block.u[j, 'ARO'].lb - parent_block.u[j, 'ARO'].ub)*disj_block.aro_10.y[j]
        return m.tox[j] - parent_block.u[j, 'ARO'] >= rhs

    return block


def _voc_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.voce = pe.Var(index_set_j(problem))
    block.vocne = pe.Var(index_set_j(problem))
    block.lin_ext_voce = pe.Var(_index_set_je(problem))

    mb = parent_block.emission_models

    voce_b = problem.attr['voce_b']
    vocne_coeff = problem.attr['vocne_coeff']
    alpha_v = problem.attr['alpha_v']
    voc_max = problem.attr['voc_max']

    @block.Constraint(_index_set_je(problem))
    def eq_lin_ext_voce(m, j, e):
        cv = problem.nodes[e].attr['cv']

        rhs = 1 + mb.e200.f[j, e] + \
            (cv[9]*mb.e300_extension.voc[j] + cv[5])*mb.aro_del.voc[j] + \
            (2*cv[8]*mb.e300_extension.voc[j] + cv[4] +
            cv[9]*mb.aro_et.voc[j])*mb.e300_del_extension.voc[j]
        return m.lin_ext_voce[j, e] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_voce(m, j):
        def _sum_row(e):
            wv = problem.nodes[e].attr['wv']
            cv = problem.nodes[e].attr['cv']
            vb = problem.nodes[e].attr['vb']
            return wv*pe.exp(
                cv[0]*parent_block.u[j, 'OXY'] +
                cv[1]*parent_block.u[j, 'SUL'] +
                cv[2]*parent_block.u[j, 'RVP'] +
                cv[3]*mb.e200_et.voc[j] +
                cv[4]*mb.e300_extension.voc[j] +
                cv[5]*mb.aro_et.voc[j] +
                cv[6]*parent_block.u[j, 'OLE'] +
                cv[7]*(mb.e200_et.voc[j]**2) +
                cv[8]*(mb.e300_extension.voc[j]**2) +
                cv[9]*mb.aro_et.voc[j]*mb.e300_extension.voc[j] - vb)*m.lin_ext_voce[j, e]
        rhs = voce_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.voce[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_vocne(m, j):
        rhs = vocne_coeff*(alpha_v[0] + alpha_v[1]*mb.rvp[j] + alpha_v[2]*(mb.rvp[j]**2))
        return m.vocne[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_constrain_voc(m, j):
        return m.voce[j] + m.vocne[j] <= voc_max

    return block


def _nox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.nox = pe.Var(index_set_j(problem))
    block.e300_nox = pe.Var(index_set_j(problem))
    block.aro_nox = pe.Var(index_set_j(problem))
    block.ole_nox = pe.Var(index_set_j(problem))
    block.lin_ext_nox = pe.Var(_index_set_je(problem))

    mb = parent_block.emission_models

    nox_b = problem.attr['nox_b']
    nox_max = problem.attr['nox_max']

    @block.Constraint(_index_set_je(problem))
    def eq_lin_ext_nox(m, j, e):
        return m.lin_ext_nox[j, e] == 1 + mb.sul.f[j, e] + mb.aro.f[j, e] + mb.ole.f[j, e]

    @block.Constraint(index_set_j(problem))
    def eq_nox(m, j):
        def _sum_row(e):
            wv = problem.nodes[e].attr['wv']
            cn = problem.nodes[e].attr['cn']
            nb = problem.nodes[e].attr['nb']
            return wv*pe.exp(
                cn[0]*parent_block.u[j, 'OXY'] +
                cn[1]*parent_block.u[j, 'SUL'] +
                cn[2]*parent_block.u[j, 'RVP'] +
                cn[3]*parent_block.u[j, 'E200'] +
                cn[4]*mb.e300_et_nox.nox[j] +
                cn[5]*mb.aro_et_nox.nox[j] +
                cn[6]*mb.ole_et_nox.nox[j] +
                cn[7]*(parent_block.u[j, 'SUL']**2) +
                cn[8]*(mb.aro_et_nox.nox[j]**2) +
                cn[9]*(mb.ole_et_nox.nox[j]**2) - nb)*m.lin_ext_nox[j, e]
        rhs = nox_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.nox[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_constrain_nox(m, j):
        return m.nox[j] <= nox_max

    return block


def _tox_block(parent_block, problem, disj_block):
    block = pe.Block()

    block.benz = pe.Var(index_set_j(problem))
    block.form = pe.Var(index_set_j(problem))
    block.acet = pe.Var(index_set_j(problem))
    block.buta = pe.Var(index_set_j(problem))
    block.nebenz = pe.Var(index_set_j(problem))

    mb = parent_block.emission_models

    benz_b = problem.attr['benz_b']
    form_b = problem.attr['form_b']
    acet_b = problem.attr['acet_b']
    buta_b = problem.attr['buta_b']
    nebenz_coeff = problem.attr['nebenz_coeff']
    alpha_nb = problem.attr['alpha_nb']
    tox_max = problem.attr['tox_max']

    @block.Constraint(index_set_j(problem))
    def eq_benz(m, j):
        def _sum_row(e):
            cbe = problem.nodes[e].attr['cbe']
            bb = problem.nodes[e].attr['bb']
            wt = problem.nodes[e].attr['wt']
            return pe.exp(
                cbe[0]*parent_block.u[j, 'OXY'] +
                cbe[1]*parent_block.u[j, 'SUL'] +
                cbe[2]*mb.e300_et_tox.tox[j] +
                cbe[3]*mb.aro_et_tox.tox[j] +
                cbe[4]*parent_block.u[j, 'BEN'] - bb)*wt
        rhs = benz_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.benz[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_form(m, j):
        def _sum_row(e):
            cf = problem.nodes[e].attr['cf']
            fb = problem.nodes[e].attr['fb']
            wt = problem.nodes[e].attr['wt']
            return pe.exp(
                cf[0]*mb.e300_et_tox.tox[j] +
                cf[1]*mb.aro_et_tox.tox[j] +
                cf[2]*parent_block.u[j, 'OLE'] +
                cf[3]*parent_block.u[j, 'MTB'] - fb)*wt
        rhs = form_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.form[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_acet(m, j):
        def _sum_row(e):
            ca = problem.nodes[e].attr['ca']
            ab = problem.nodes[e].attr['ab']
            wt = problem.nodes[e].attr['wt']
            return pe.exp(
                ca[0]*parent_block.u[j, 'SUL'] +
                ca[1]*parent_block.u[j, 'RVP'] +
                ca[2]*mb.e300_et_tox.tox[j] +
                ca[3]*mb.aro_et_tox.tox[j] +
                ca[4]*parent_block.u[j, 'MTB'] +
                ca[5]*parent_block.u[j, 'ETB'] +
                ca[6]*parent_block.u[j, 'ETH'] - ab)*wt
        rhs = acet_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.acet[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_buta(m, j):
        def _sum_row(e):
            cbu = problem.nodes[e].attr['cbu']
            db = problem.nodes[e].attr['db']
            wt = problem.nodes[e].attr['wt']
            return pe.exp(
                cbu[0]*parent_block.u[j, 'OXY'] +
                cbu[1]*parent_block.u[j, 'SUL'] +
                cbu[2]*parent_block.u[j, 'E200'] +
                cbu[3]*mb.e300_et_tox.tox[j] +
                cbu[4]*mb.aro_et_tox.tox[j] +
                cbu[5]*parent_block.u[j, 'OLE'] - db)*wt
        rhs = buta_b*sum(_sum_row(e) for e in index_set_j(problem))
        return m.buta[j] == rhs

    @block.Constraint(index_set_j(problem))
    def eq_nebenz(m, j):
        rhs = (
                alpha_nb[0]*parent_block.u[j, 'BEN'] +
                alpha_nb[1]*parent_block.u[j, 'RVP']*parent_block.u[j, 'BEN'] +
                alpha_nb[2]*parent_block.u[j, 'MTB']*parent_block.u[j, 'BEN'] +
                alpha_nb[3]*parent_block.u[j, 'RVP']*parent_block.u[j, 'RVP']*parent_block.u[j, 'BEN'] +
                alpha_nb[4]*parent_block.u[j, 'RVP']*parent_block.u[j, 'MTB']*parent_block.u[j, 'BEN'] +
                alpha_nb[5]*parent_block.u[j, 'RVP']*parent_block.u[j, 'RVP']*parent_block.u[j, 'RVP']*parent_block.u[j, 'BEN'] +
                alpha_nb[6]*parent_block.u[j, 'RVP']*parent_block.u[j, 'RVP']*parent_block.u[j, 'MTB']*parent_block.u[j, 'BEN']
        )
        return m.nebenz[j] == nebenz_coeff*rhs

    @block.Constraint(index_set_j(problem))
    def eq_tox(m, j):
        return m.benz[j] + m.form[j] + m.acet[j] + m.buta[j] + m.nebenz[j] + 0.003355*mb.voc.voce[j] <= tox_max

    return block


def _f_sul_bounds(parent_block: pe.Block, problem: Network):
    def _bounds(m, j, e):
        cn = problem.nodes[e].attr['cn']
        lb = np.min([
            0,
            (2*cn[7]*10 + cn[1])*(parent_block.u[j, 'SUL'].lb - 10),
            (2*cn[7]*450 + cn[1])*(parent_block.u[j, 'SUL'].lb - 450),
            (2*cn[7]*10 + cn[1])*(parent_block.u[j, 'SUL'].ub - 10),
            (2*cn[7]*450 + cn[1])*(parent_block.u[j, 'SUL'].ub - 450),
        ])
        ub = np.max([
            0,
            (2*cn[7]*10 + cn[1])*(parent_block.u[j, 'SUL'].lb - 10),
            (2*cn[7]*450 + cn[1])*(parent_block.u[j, 'SUL'].lb - 450),
            (2*cn[7]*10 + cn[1])*(parent_block.u[j, 'SUL'].ub - 10),
            (2*cn[7]*450 + cn[1])*(parent_block.u[j, 'SUL'].ub - 450),
        ])
        return lb, ub
    return _bounds


def _f_aro_bounds(parent_block: pe.Block, problem: Network):
    def _bounds(m, j, e):
        cn = problem.nodes[e].attr['cn']
        lb = np.min([
            0,
            (2*cn[8]*18 + cn[5])*(parent_block.u[j, 'ARO'].lb - 18),
            (2*cn[8]*18 + cn[5])*(-8),
            (2*cn[8]*18 + cn[5])*(parent_block.u[j, 'ARO'].ub - 18),
        ])
        ub = np.max([
            0,
            (2*cn[8]*18 + cn[5])*(parent_block.u[j, 'ARO'].lb - 18),
            (2*cn[8]*18 + cn[5])*(-8),
            (2*cn[8]*18 + cn[5])*(parent_block.u[j, 'ARO'].ub - 18),
        ])
        return lb, ub
    return _bounds


def _f_ole_bounds(parent_block: pe.Block, problem: Network):
    def _bounds(m, j, e):
        cn = problem.nodes[e].attr['cn']
        lb = np.min([
            0,
            (2*cn[9]*19 + cn[6])*(parent_block.u[j, 'OLE'].lb - 19),
            (2*cn[9]*19 + cn[6])*(parent_block.u[j, 'OLE'].ub - 19),
        ])
        ub = np.max([
            0,
            (2*cn[9]*19 + cn[6])*(parent_block.u[j, 'OLE'].lb - 19),
            (2*cn[9]*19 + cn[6])*(parent_block.u[j, 'OLE'].ub - 19),
        ])
        return lb, ub
    return _bounds


def _f_e200_bounds(parent_block: pe.Block, problem: Network):
    def _bounds(m, j, e):
        cv = problem.nodes[e].attr['cv']
        lb = np.min([
            0,
            (2*cv[7]*33 + cv[3])*(parent_block.u[j, 'E200'].lb - 33),
            (2*cv[7]*33 + cv[3])*(parent_block.u[j, 'E200'].ub - 33),
        ])
        ub = np.max([
            0,
            (2*cv[7]*33 + cv[3])*(parent_block.u[j, 'E200'].lb - 33),
            (2*cv[7]*33 + cv[3])*(parent_block.u[j, 'E200'].ub - 33),
        ])
        return lb, ub
    return _bounds


def rvp_bounds(parent_block: pe.Block, problem: Network):
    def _bounds(m, j):
        return parent_block.u[j, 'RVP'].bounds
    return _bounds


def _e200_voc_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        e200_lb, e200_ub = parent_block.u[j, 'E200'].bounds
        lb = e200_lb
        ub = e200_ub
        lb = min(65.52, max(33, lb))
        ub = min(65.52, max(33, ub))
        return lb, ub
    return _bounds


def _aro_voc_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        lb, ub = parent_block.u[j, 'ARO'].bounds
        lb = min(46, max(18, lb))
        ub = min(46, max(18, ub))
        return lb, ub
    return _bounds


def _aro_voc_del_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        aro_lb, aro_ub = parent_block.u[j, 'ARO'].bounds
        lb = ub = 0.0
        if aro_lb < 18.0:
            lb = aro_lb - 18.0
        if aro_ub < 18:
            ub = aro_ub - 18.0
        if aro_lb > 46:
            lb = aro_lb - 46.0
        if aro_ub > 46:
            ub = aro_ub - 46.0
        if lb < -8:
            lb = -8
        if ub < -8:
            ub = -8
        return lb, ub
    return _bounds


def _e300_voc_star_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        lb = 0.385*10 + 79.75
        ub = 0.385*parent_block.u[j, 'ARO'].ub + 79.75
        return lb, ub
    return _bounds


def _e300_voc_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        e30_lb, e30_ub = parent_block.u[j, 'E300'].bounds
        lb = e30_lb
        ub = e30_ub
        if e30_lb < 72:
            lb = 72
        if e30_ub < 72:
            ub = 72
        if e30_lb > 94:
            lb = 94
        if e30_ub > 94:
            ub = 94
        return lb, ub
    return _bounds


def _e300_voc_del_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        e30_lb, e30_ub = parent_block.u[j, 'E300'].bounds
        lb = ub = 0.0
        if e30_lb < 72:
            lb = e30_lb - 72
        if e30_ub < 72:
            ub = e30_ub - 72
        if e30_lb > 94:
            lb = min(e30_lb - 94, 1)
        if e30_ub > 94:
            ub = min(e30_ub - 94, 1)
        return lb, ub
    return _bounds


def _sul_nox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        sul_lb, sul_ub = parent_block.u[j, 'SUL'].bounds
        lb = sul_lb
        ub = sul_ub
        if sul_lb < 10:
            lb = 10
        if sul_ub < 10:
            ub = 10
        if sul_lb > 450:
            lb = 450
        if sul_ub > 450:
            ub = 450
        return lb, ub
    return _bounds


def _e300_nox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        sul_lb, sul_ub = parent_block.u[j, 'E300'].bounds
        lb = sul_lb
        ub = sul_ub
        if sul_lb > 95:
            lb = 95
        if sul_ub > 95:
            ub = 95
        return lb, ub
    return _bounds


def _aro_nox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        aro_lb, aro_ub = parent_block.u[j, 'ARO'].bounds
        lb = aro_lb
        ub = aro_ub
        if aro_lb < 18:
            lb = 18
        if aro_ub < 18:
            ub = 18
        if aro_lb > 36.8:
            lb = 36.8
        if aro_ub > 36.8:
            ub = 36.8
        return lb, ub
    return _bounds


def _ole_nox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        ole_lb, ole_ub = parent_block.u[j, 'OLE'].bounds
        lb = ole_lb
        ub = ole_ub
        if ole_lb < 3.77:
            lb = 3.77
        if ole_ub < 3.77:
            ub = 3.77
        if ole_lb > 19:
            lb = 19.0
        if ole_ub > 19:
            ub = 19.0
        return lb, ub
    return _bounds


def _e300_tox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        e300_lb, e300_ub = parent_block.u[j, 'E300'].bounds
        lb = e300_lb
        ub = e300_ub
        if e300_lb > 95:
            lb = 95.0
        if e300_ub > 95:
            ub = 95.0
        return lb, ub
    return _bounds


def _aro_tox_bounds(parent_block: pe.Block):
    def _bounds(m, j):
        aro_lb, aro_ub = parent_block.u[j, 'ARO'].bounds
        lb = aro_lb
        ub = aro_ub
        if aro_lb < 10:
            lb = 10.0
        if aro_ub < 10:
            ub = 10.0
        return lb, ub
    return _bounds
