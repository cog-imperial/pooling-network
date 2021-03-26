import pyomo.environ as pe

from pooling_network.formulation.pq_block import PoolingPQFormulation
from pooling_network.formulation.pq import pooling_problem_pq_formulation
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.instances.literature import literature_problem_data


# Make two copies of the pooling problem
problems = [
    pooling_problem_from_data(literature_problem_data('adhya4')),
    pooling_problem_from_data(literature_problem_data('adhya4')),
]


m = pe.ConcreteModel()


# Option 1: create and return a PoolingPQFormulation block
@m.Block(range(2))
def pooling_problem(b, i):
    pooling = PoolingPQFormulation()
    pooling.set_pooling_problem(problems[i])
    pooling.rebuild()
    return pooling


# Option 2: add the constraints to the block passed by Pyomo
@m.Block(range(2))
def pooling_problem_opt_2(b, i):
    pooling_problem_pq_formulation(b, problems[i])
    # Notice no return here.


m.pprint()