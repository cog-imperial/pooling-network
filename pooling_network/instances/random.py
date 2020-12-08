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
import requests
import json
import networkx as nx
from pooling_network.network import Node, Edge, Network


_DEFAULT_REPOSITORY = 'poolinginstances/poolinginstances'


def download_random_instance_data(name, repository=None):
    if repository is None:
        repository = _DEFAULT_REPOSITORY
    url = 'https://raw.githubusercontent.com/{repo}/master/data/random_haverly/{name}.json'.format(
        repo=repository,
        name=name,
    )

    response = requests.get(url)

    if response.status_code == 200:
        return json.loads(response.content)

    raise RuntimeError('Request returned non-success status code: {} {}'.format(response.status_code, response.content))


def build_problem_from_instance_data(data, name):
    graph_data = data['graph']

    nodes = []
    problem = Network(name)

    for node in graph_data['nodes']:
        node_type = node['type']
        node_id = node['id'].replace('_', '-')
        if node_type == 'input':
            node = problem.add_node(
                layer=0,
                name=node_id,
                cost=0.0,
                capacity_lower=None,
                capacity_upper=node['C'],
                attr={
                    'quality': node['lambda']
                }
            )
            nodes.append(node)
        elif node_type == 'output':
            node = problem.add_node(
                layer=2,
                name=node_id,
                cost=0.0,
                capacity_lower=None,
                capacity_upper=node['C'],
                attr={
                    'quality_upper': node['overbeta']
                }
            )
            nodes.append(node)
        elif node_type == 'pool':
            node = problem.add_node(
                layer=1,
                name=node_id,
                cost=0.0,
                capacity_lower=None,
                capacity_upper=node['C'],
                attr={}
            )
            nodes.append(node)
        else:
            raise ValueError('Invalid node type {}'.format(node_type))

    for edge in graph_data['links']:
        source = edge['source']
        target = edge['target']
        cost = edge['cost']
        source = nodes[source]
        target = nodes[target]

        if target.layer == 1:
            limit = 1.0
        elif target.layer == 2:
            _, limit = target.capacity
        else:
            raise ValueError('Invalid target type {}: {}'.format(type(target), target))
        problem.add_edge(source.name, target.name, cost=cost, capacity_upper=limit)

    return problem


def random_instance(name, repository=None):
    data = download_random_instance_data(name, repository)
    return build_problem_from_instance_data(data, name)


def _random_instances_names(i, size, ds, attributes):
    for d in ds:
        for at in attributes:
            yield 'haverly_{size}_addedges_{d}_attr_{at}_{i}'.format(
                size=size,
                d=d,
                at=at,
                i=i
            )


def random_instances_names():
    """Iterator to generate random instances names.

    References:
    https://github.com/poolinginstances/poolinginstances/blob/master/scripts/genrandom_addedges.sh
    """
    attributes = [0, 1]
    for i in range(1, 11):
        yield from _random_instances_names(i, 10, [10, 20, 30, 40, 50, 60], attributes)
        yield from _random_instances_names(i, 15, [15, 30, 45, 60, 75, 90], attributes)
        yield from _random_instances_names(i, 20, [20, 40, 60, 80, 100, 120], attributes)
