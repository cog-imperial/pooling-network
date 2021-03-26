import pyomo.environ as pe
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.formulation.pq_block import PoolingPQFormulation



def get_pyomo_model(*args, **kwargs):
    name = 'adhya4'

    print('Problem ', name)
    problem = pooling_problem_from_data(literature_problem_data(name))

    model = pe.ConcreteModel()

    model.pooling = PoolingPQFormulation()
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()

    model.pooling.add_objective(use_flow_cost=False)

    return model
