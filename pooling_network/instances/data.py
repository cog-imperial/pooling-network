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
from typing import Dict, NamedTuple, Tuple, Optional

from pooling_network.network import (
    Network,
)


QualityDict = Dict[str, float]


class ComponentData(NamedTuple):
    lower: float
    upper: float
    price: float
    quality: QualityDict


class ProductData(NamedTuple):
    lower: float
    upper: float
    price: float
    quality_lower: Optional[QualityDict]
    quality_upper: QualityDict


class PoolingProblemData(NamedTuple):
    name: str
    optimal_objective: Optional[float]
    component: Dict[str, ComponentData]
    product: Dict[str, ProductData]
    pool_size: Dict[str, float]
    component_pool_fraction: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]
    pool_product_bound: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]
    component_product_bound: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]


def pooling_problem_from_data(pooling_data: PoolingProblemData) -> Network:
    network = Network(pooling_data.name)

    for name, data in pooling_data.component.items():
        network.add_node(
            layer=0,
            name=name,
            capacity_lower=data.lower,
            capacity_upper=data.upper,
            cost=data.price,
            attr={
                'quality': data.quality,
            }
        )

    for name, data in pooling_data.product.items():
        network.add_node(
            layer=2,
            name=name,
            capacity_lower=data.lower,
            capacity_upper=data.upper,
            cost=data.price,
            attr={
                'quality_lower': data.quality_lower,
                'quality_upper': data.quality_upper,
            }
        )

    for name, size in pooling_data.pool_size.items():
        network.add_node(
            layer=1,
            name=name,
            capacity_lower=None,
            capacity_upper=size,
            cost=0.0,
            attr={},
        )

    for (from_, to), (limit, cost) in pooling_data.component_pool_fraction.items():
        if limit is None:
            continue
        network.add_edge(from_, to, capacity_upper=limit, cost=cost)

    for (from_, to), (limit, cost) in pooling_data.pool_product_bound.items():
        if limit is None:
            continue
        network.add_edge(from_, to, capacity_upper=limit, cost=cost)

    for (from_, to), (limit, cost) in pooling_data.component_product_bound.items():
        if limit is None:
            continue
        network.add_edge(from_, to, capacity_upper=limit, cost=cost)

    network.attr['known_optimal_objective'] = pooling_data.optimal_objective

    return network
