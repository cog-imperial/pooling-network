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
from typing import Any, Dict, NamedTuple, List, Tuple, Optional

import networkx as nx


class Node(NamedTuple):
    name: str
    layer: int
    capacity: Tuple[Optional[float], Optional[float]]
    cost: float
    attr: Dict[str, Any]

    @property
    def capacity_lower(self):
        return self.capacity[0]

    @property
    def capacity_upper(self):
        return self.capacity[1]


class Edge(NamedTuple):
    source: Node
    destination: Node
    cost: float
    fixed_cost: float
    capacity: Tuple[Optional[float], Optional[float]]
    attr: Dict[str, Any]

    @property
    def capacity_lower(self):
        return self.capacity[0]

    @property
    def capacity_upper(self):
        return self.capacity[1]


class _NodeDict:
    def __init__(self, nodes):
        self._nodes = nodes

    def __getitem__(self, name):
        return self._nodes[name]


class _EdgeDict:
    def __init__(self, nodes, graph):
        self._nodes = nodes
        self._graph = graph

    def __getitem__(self, edge):
        src, dest = edge
        try:
            graph_edge = self._graph[src][dest]
            edge_attr = graph_edge['attr']
            source = self._nodes[src]
            destination = self._nodes[dest]
            return Edge(
                source=source,
                destination=destination,
                capacity=edge_attr['capacity'],
                cost=edge_attr['cost'],
                fixed_cost=edge_attr['fixed_cost'],
                attr=edge_attr['attr']
            )
        except KeyError:
            raise KeyError('{}, {}'.format(src, dest))


class Network:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._nodes_by_name: Dict[str, Node] = dict()
        self._nodes_by_layer: Dict[int, List[Node]] = dict()
        self._graph = nx.DiGraph()
        self.attr = dict()

    def add_node(self, layer: int, name: str, capacity_lower: Optional[float], capacity_upper: Optional[float],
                 cost: float, attr: Dict[str, Any]):
        node = Node(name, layer, capacity=(capacity_lower, capacity_upper), cost=cost, attr=attr)
        self._nodes_by_name[node.name] = node
        if layer not in self._nodes_by_layer:
            self._nodes_by_layer[layer] = []
        self._nodes_by_layer[layer].append(node)
        self._graph.add_node(node.name)
        return node

    def add_edge(self, src: str, dest: str, capacity_lower: Optional[float] = None, capacity_upper: Optional[float] = None,
                 cost: float = 0.0, fixed_cost: float = 0.0, attr: Optional[Dict[str, Any]] = None):
        if attr is None:
            attr = dict()

        edge_attr = {
            'attr': attr,
            'capacity': (capacity_lower, capacity_upper),
            'cost': cost,
            'fixed_cost': fixed_cost,
        }

        self._graph.add_edge(src, dest, attr=edge_attr)

    def successors(self, src: str, layer: Optional[int] = None):
        yield from self._filter_nodes(self._graph.successors(src), layer)

    def predecessors(self, src: str, layer: Optional[int] = None):
        yield from self._filter_nodes(self._graph.predecessors(src), layer)

    def _filter_nodes(self, node_gen, layer):
        for node_name in node_gen:
            node = self._nodes_by_name[node_name]
            if layer is not None and node.layer != layer:
                continue
            yield node

    @property
    def nodes(self):
        return _NodeDict(self._nodes_by_name)

    def nodes_at_layer(self, layer):
        return self._nodes_by_layer[layer]

    @property
    def edges(self):
        return _EdgeDict(self._nodes_by_name, self._graph)
