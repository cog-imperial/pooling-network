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
import json
from typing import Union
from pathlib import Path
from pooling_network.instances.data import (
    PoolingProblemData,
    ComponentData,
    ProductData,
)


literature_instances = [
    'adhya1', 'adhya2', 'adhya3', 'adhya4',
    'bental4', 'bental5',
    'foulds2', 'foulds3', 'foulds4', 'foulds5',
    'haverly1', 'haverly2', 'haverly3', 'rt2'
]

randstd_instances = ['randstd{}'.format(n) for n in range(11, 61)]


def pooling_problem_data_from_data(data: dict) -> PoolingProblemData:
    """Convert unstructured data to structured pooling problem data"""
    components = dict()
    for comp_data in data['components']:
        comp_name = comp_data['name']
        components[comp_name] = ComponentData(
            lower=comp_data['lower'],
            upper=comp_data['upper'],
            price=comp_data['price'],
            quality=comp_data['quality']
        )

    products = dict()
    for prod_data in data['products']:
        prod_name = prod_data['name']
        products[prod_name] = ProductData(
            lower=prod_data['lower'],
            upper=prod_data['upper'],
            price=prod_data['price'],
            quality_lower=prod_data['quality_lower'],
            quality_upper=prod_data['quality_upper'],
        )

    pool_size = data['pool_size']

    component_pool_fraction = dict()
    for con_data in data['component_to_pool_fraction']:
        component_pool_fraction[(con_data['component'], con_data['pool'])] = \
            (con_data['fraction'], con_data.get('cost'))

    pool_product_bound = dict()
    for con_data in data['pool_to_product_bound']:
        pool_product_bound[(con_data['pool'], con_data['product'])] = \
            (con_data['bound'], con_data.get('cost'))

    component_product_bound = dict()
    for con_data in data['component_to_product_bound']:
        component_product_bound[(con_data['component'], con_data['product'])] = \
            (con_data['bound'], con_data.get('cost'))

    return PoolingProblemData(
        name=data['name'],
        optimal_objective=data.get('objective'),
        component=components,
        product=products,
        pool_size=pool_size,
        component_pool_fraction=component_pool_fraction,
        pool_product_bound=pool_product_bound,
        component_product_bound=component_product_bound,
    )


def pooling_problem_data_from_json(filename: Union[str, Path]) -> PoolingProblemData:
    """Read pooling problem data from the json file"""
    with open(filename) as f:
        data = json.load(f)
        return pooling_problem_data_from_data(data)


def literature_problem_data(name: str) -> PoolingProblemData:
    return pooling_problem_data_from_json(_data_file_path(name))


def _data_file_path(problem_name: str) -> Path:
    current_dir = Path(__file__).parent
    filename = problem_name + '.json'
    return (current_dir / 'data' / filename).resolve()