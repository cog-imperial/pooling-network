import numpy as np
import pyomo.environ as pe
from galini.relaxations.relax import relax, RelaxationData

from pooling_network.block import PoolingPQFormulation
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.instances.literature import literature_problem_data
from pooling_network.pooling import (
    problem_pool_output_qualities, compute_gamma_ijk, compute_beta_kl_bounds, compute_gamma_kl_bounds, index_set_ilj
)


def test_haverly1():
    # Solvers setup
    mip_solver = pe.SolverFactory('cplex_direct')
    global_solver = pe.SolverFactory('gams')
    global_solver.options['solver'] = 'baron'
    global_solver.options['tee'] = False

    # Build PQ-formulation
    problem = pooling_problem_from_data(literature_problem_data('haverly1'))
    model = pe.ConcreteModel()

    model.pooling = PoolingPQFormulation()
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()

    objective = model.pooling.add_objective(use_flow_cost=False)

    # Solve globally with gams to check the model is correct
    global_solver.solve(model)
    assert pe.value(objective) == -400.0

    # Build linear relaxation and solve it
    relaxation_data = RelaxationData(model)
    relaxed_model = relax(model, relaxation_data)
    relaxed_objective = relaxed_model.find_component(objective.getname(fully_qualified=True))
    mip_solver.solve(relaxed_model)

    # Compute error between quadratic formulation and its relaxation
    expected_err = {
        ('c1', 'o1', 'p1'): 25.0,
        ('c1', 'o1', 'p2'): 50.0,
        ('c2', 'o1', 'p1'): 25.0,
        ('c2', 'o1', 'p2'): 50.0,
    }
    for i, l, j in index_set_ilj(problem):
        err_expr = abs(
            relaxed_model.pooling.v[i, l, j] - relaxed_model.pooling.q[i, l] * relaxed_model.pooling.y[l, j]
        )
        err = pe.value(err_expr)
        np.testing.assert_almost_equal(err, expected_err[i, l, j])

    # Check qualities
    expected_gamma = {
        ('c1', 'p1', 'q1'): 0.5,
        ('c2', 'p1', 'q1'): -1.5,
        ('c3', 'p1', 'q1'): -0.5,
        ('c1', 'p2', 'q1'): 1.5,
        ('c2', 'p2', 'q1'): -0.5,
        ('c3', 'p2', 'q1'): 0.5,
    }
    for input in problem.nodes_at_layer(0):
        for output in problem.successors(input.name, layer=2):
            for k, q in output.attr['quality_upper'].items():
                gamma = compute_gamma_ijk(input, output, k)
                np.testing.assert_almost_equal(gamma, expected_gamma[input.name, output.name, k])

    expected_gamma_lower = {
        ('o1', 'p1', 'q1'): -1.5,
        ('o1', 'p2', 'q1'): -0.5,
    }
    expected_gamma_upper = {
        ('o1', 'p1', 'q1'): 0.5,
        ('o1', 'p2', 'q1'): 1.5,
    }
    expected_beta_lower = {
        ('o1', 'p1', 'q1'): -0.5,
        ('o1', 'p2', 'q1'): 0.5,
    }
    expected_beta_upper = {
        ('o1', 'p1', 'q1'): -0.5,
        ('o1', 'p2', 'q1'): 0.5,
    }
    for l, j, k in problem_pool_output_qualities(problem):
        gamma_lower, gamma_upper = compute_gamma_kl_bounds(l, j, k, problem)
        beta_lower, beta_upper = compute_beta_kl_bounds(l, j, k, problem)

        np.testing.assert_almost_equal(gamma_lower, expected_gamma_lower[l, j, k])
        np.testing.assert_almost_equal(gamma_upper, expected_gamma_upper[l, j, k])
        np.testing.assert_almost_equal(beta_lower, expected_beta_lower[l, j, k])
        np.testing.assert_almost_equal(beta_upper, expected_beta_upper[l, j, k])

    # Now add variables inequalities only
    relaxed_model.pooling.add_inequalities(add_inequalities=False, add_uxt=True)
    mip_solver.solve(relaxed_model)
    np.testing.assert_almost_equal(-500.0, pe.value(relaxed_objective))

    ineq_block = relaxed_model.pooling.inequalities
    # Check the value of variables after solve
    expected_z = {
        ('o1', 'p1', 'q1'): 0.5,
        ('o1', 'p2', 'q1'): 0.5,
    }
    expected_t = {
        ('o1', 'p1', 'q1'): -0.5,
        ('o1', 'p2', 'q1'): 0.5,
    }
    expected_u = {
        ('o1', 'p1', 'q1'): 0.25,
        ('o1', 'p2', 'q1'): -0.25,
    }
    expected_y = {
        ('o1', 'p1', 'q1'): -0.25,
        ('o1', 'p2', 'q1'): 0.25,
    }
    expected_s = {
        ('o1', 'p1'): 0.5,
        ('o1', 'p2'): 0.5,
    }
    for l, j, k in problem_pool_output_qualities(problem):
        np.testing.assert_almost_equal(ineq_block.z[l, j].value, expected_z[l, j, k])
        np.testing.assert_almost_equal(ineq_block.t[j, k, l].value, expected_t[l, j, k])
        np.testing.assert_almost_equal(ineq_block.u[j, k, l].value, expected_u[l, j, k])
        np.testing.assert_almost_equal(ineq_block.y[j, k, l].value, expected_y[l, j, k])
        np.testing.assert_almost_equal(ineq_block.s[l, j].value, expected_s[l, j])

    # Now add everything.
    relaxed_model.pooling.add_inequalities(add_inequalities=True, add_uxt=True)
    assert 2 == len(relaxed_model.pooling.inequalities._inequalities)
    mip_solver.solve(relaxed_model)
    np.testing.assert_almost_equal(pe.value(relaxed_objective), -500.0)

    # Now add the cuts, in the first iteration it will add Equation 15 and 18.
    relaxed_model.pooling.add_cuts()
    assert 2 == len(relaxed_model.pooling.inequalities._cuts)
    mip_solver.solve(relaxed_model)
    relaxed_model.pooling.add_cuts()
    # No cuts added
    assert 2 == len(relaxed_model.pooling.inequalities._cuts)
    np.testing.assert_almost_equal(pe.value(relaxed_objective), -400.0)