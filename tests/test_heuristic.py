import numpy as np
import pyomo.environ as pe

from pooling_network.formulation.pq_block import PoolingPQFormulation
from pooling_network.heuristic import (
    add_mip_heuristic,
    derive_fractional_flow_variables,
)
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.instances.literature import literature_problem_data


def test_mip_heuristic():
    problem = pooling_problem_from_data(literature_problem_data('adhya4'))

    model = pe.ConcreteModel()

    model.pooling = PoolingPQFormulation()
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()

    model.pooling.add_objective(use_flow_cost=False)

    mip_solver = pe.SolverFactory('cplex')

    add_mip_heuristic(model.pooling, problem, tau=2)
    mip_solver.solve(model)
    derive_fractional_flow_variables(model.pooling)

    assert np.isclose(-813.1224489795917, pe.value(model.pooling.cost))
    for v in model.component_data_objects(pe.Var):
        assert pe.value(v) is not None
