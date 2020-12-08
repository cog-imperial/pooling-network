import pytest
from pooling_network.network import Node, Edge, Network


def test_network():
    n = Network('test')

    n.add_node(
        layer=0,
        name='n0',
        capacity_lower=None,
        capacity_upper=100.0,
        cost=10,
        attr={
            'quality_upper': {
                'q1': 1.0
            }
        }
    )
    n.add_node(
        layer=0,
        name='n1',
        capacity_lower=None,
        capacity_upper=10.0,
        cost=9.0,
        attr={
            'quality_upper': {
                'q1': 2.0
            }
        }
    )
    n.add_node(
        layer=0,
        name='n2',
        capacity_lower=None,
        capacity_upper=30.0,
        cost=-4.9,
        attr={
            'quality_upper': {
                'q1': 0.5
            }
        }
    )

    n.add_node(
        layer=1,
        name='p2',
        capacity_lower=None,
        capacity_upper=30.0,
        cost=0.0,
        attr={
            'quality_upper': {
                'q1': 0.5
            }
        }
    )

    n.add_node(
        layer=2,
        name='p3',
        capacity_lower=None,
        capacity_upper=30.0,
        cost=1.0,
        attr={
            'quality_upper': {
                'q1': 0.5
            }
        }
    )

    n.add_edge('n1', 'p2', capacity_upper=123.4)
    n.add_edge('n1', 'p3')

    edge = n.edges['n1', 'p2']
    assert edge.capacity == (None, 123.4)
    assert edge.attr.get('attr') is None

    assert isinstance(n.nodes['n2'], Node)

    succ = n.successors('n1')
    assert len(list(succ)) == 2

    succ_filter = n.successors('n1', 1)
    assert len(list(succ_filter)) == 1

    assert len(list(n.predecessors('n1'))) == 0
    assert len(list(n.predecessors('p2', 0))) == 1
