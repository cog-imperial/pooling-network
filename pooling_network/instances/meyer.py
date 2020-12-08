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
from pooling_network.network import Network


def meyer4() -> Network:
    pool_flow_cost = {
        'P1': 1102.9,
        'P2': 2895.2,
        'P3': 1102.9,
        'P4': 1102.9,
    }

    pool_fixed_cost = {
        'P1': 13972,
        'P2': 36676,
        'P3': 13972,
        'P4': 13972,
    }

    distance_input_output = {
        'I1': {'O1': 150},
        'I2': {'O1': 135},
        'I3': {'O1': 100},
        'I4': {'O1': 90},
        'I5': {'O1': 40},
        'I6': {'O1': 70},
        'I7': {'O1': 45},
    }

    distance_pool_output = {
        'P1': {'O1': 75},
        'P2': {'O1': 80},
        'P3': {'O1': 160},
        'P4': {'O1': 190},
    }

    distance_pool_pool = {
        'P1': {'P2': 80, 'P3': 210, 'P4': 190},
        'P2': {'P1': 80, 'P3': 130, 'P4': 100},
        'P3': {'P1': 210, 'P2': 130, 'P4': 110},
        'P4': {'P1': 190, 'P2': 100, 'P3': 110},
    }

    distance_input_pool = {
        'I1': {'P1': 75, 'P2': 150, 'P3': 280, 'P4': 245},
        'I2': {'P1': 55, 'P2': 125, 'P3': 260, 'P4': 215},
        'I3': {'P1': 30, 'P2': 115, 'P3': 240, 'P4': 220},
        'I4': {'P1': 55, 'P2': 140, 'P3': 245, 'P4': 245},
        'I5': {'P1': 55, 'P2': 40, 'P3': 150, 'P4': 150},
        'I6': {'P1': 40, 'P2': 120, 'P3': 230, 'P4': 230},
        'I7': {'P1': 30, 'P2': 60, 'P3': 175, 'P4': 165},
    }

    problem = Network('scheduling')

    problem.add_node(
        layer=0,
        name='I1',
        capacity_lower=20.0,
        capacity_upper=20.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 100,
                'Q2': 500,
                'Q3': 500,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I2',
        capacity_lower=50.0,
        capacity_upper=50.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 800,
                'Q2': 1750,
                'Q3': 2000,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I3',
        capacity_lower=47.5,
        capacity_upper=47.5,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 400,
                'Q2': 80,
                'Q3': 100,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I4',
        capacity_lower=28.0,
        capacity_upper=28.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 1200,
                'Q2': 1000,
                'Q3': 400,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I5',
        capacity_lower=100.0,
        capacity_upper=100.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 500,
                'Q2': 700,
                'Q3': 250,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I6',
        capacity_lower=30.0,
        capacity_upper=30.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 50,
                'Q2': 100,
                'Q3': 50,
            }
        },
    )

    problem.add_node(
        layer=0,
        name='I7',
        capacity_lower=25.0,
        capacity_upper=25.0,
        cost=0.0,
        attr={
            'quality': {
                'Q1': 1000,
                'Q2': 50,
                'Q3': 150,
            }
        },
    )

    problem.add_node(
        layer=1,
        name='P1',
        capacity_lower=0.0,
        capacity_upper=300.5,
        cost=0.0,
        attr={
            'reduction_ratio': {
                'Q1': 0.99,
                'Q2': 0.90,
                'Q3': 0.95,
            }
        }
    )

    problem.add_node(
        layer=1,
        name='P2',
        capacity_lower=0.0,
        capacity_upper=300.5,
        cost=0.0,
        attr={
            'reduction_ratio': {
                'Q1': 0.00,
                'Q2': 0.87,
                'Q3': 0.90,
            }
        }
    )

    problem.add_node(
        layer=1,
        name='P3',
        capacity_lower=0.0,
        capacity_upper=300.5,
        cost=0.0,
        attr={
            'reduction_ratio': {
                'Q1': 0.10,
                'Q2': 0.99,
                'Q3': 0.00,
            }
        }
    )

    problem.add_node(
        layer=1,
        name='P4',
        capacity_lower=0.0,
        capacity_upper=300.5,
        cost=0.0,
        attr={
            'reduction_ratio': {
                'Q1': 0.70,
                'Q2': 0.20,
                'Q3': 0.30,
            }
        }
    )

    problem.add_node(
        layer=2,
        name='O1',
        capacity_lower=300.5,
        capacity_upper=300.5,
        cost=0.0,
        attr={
            'quality_lower': {
                'Q1': 0.0,
                'Q2': 0.0,
                'Q3': 0.0,
            },
            'quality_upper': {
                'Q1': 5.0,
                'Q2': 5.0,
                'Q3': 10.0,
            }
        },
    )

    c_1 = 3603.4
    c_2 = 124.6
    nu = 3600

    _add_edges_from_table(problem, distance_input_pool, c_1, c_2, nu)
    _add_edges_from_table(problem, distance_pool_output, c_1, c_2, nu)
    _add_edges_from_table(problem, distance_pool_pool, c_1, c_2, nu)
    _add_edges_from_table(problem, distance_input_output, c_1, c_2, nu)

    problem.attr['fmin'] = 0.2

    return problem


def _add_edges_from_table(problem, table, c_1, c_2, nu):
    for from_idx, dest_dist in table.items():
        for to_idx, dist in dest_dist.items():
            cost = dist * (c_1 / nu)
            fixed_cost = cost * c_2
            problem.add_edge(from_idx, to_idx, cost=cost, fixed_cost=fixed_cost)
