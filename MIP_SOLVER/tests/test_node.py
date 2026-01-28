"""
Unit tests for the Node class (solver/node.py).

Tests the node comparison logic for different search strategies:
- Depth-First Search (DFS): deeper nodes have higher priority
- Best-Bound: nodes with better LP objectives have higher priority
"""
import pytest
from solver.node import Node


class TestNodeComparator:
    """Tests for the Node.__lt__ comparator method."""

    def setup_method(self):
        """Reset Node class variables before each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def teardown_method(self):
        """Reset Node class variables after each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def test_dfs_deeper_node_has_priority(self):
        """In DFS mode, deeper nodes should be selected first (lower in heap)."""
        shallow_node = Node(node_id=1, parent_id=0, lp_objective=100.0, depth=2)
        deep_node = Node(node_id=2, parent_id=1, lp_objective=100.0, depth=5)

        # In a min-heap, __lt__ returning True means higher priority
        # DFS: deeper nodes should have higher priority (return True)
        assert deep_node < shallow_node
        assert not shallow_node < deep_node

    def test_dfs_same_depth_comparison(self):
        """In DFS mode, nodes at the same depth should not have priority over each other."""
        node1 = Node(node_id=1, parent_id=0, lp_objective=100.0, depth=3)
        node2 = Node(node_id=2, parent_id=0, lp_objective=50.0, depth=3)

        # Same depth means neither has priority over the other
        assert not node1 < node2
        assert not node2 < node1

    def test_best_bound_minimization(self):
        """In Best-Bound mode for minimization, lower LP objective has priority."""
        Node.switch_to_bb = True
        Node.is_maximization = False

        better_node = Node(node_id=1, parent_id=0, lp_objective=50.0, depth=2)
        worse_node = Node(node_id=2, parent_id=0, lp_objective=100.0, depth=5)

        # For minimization, lower objective is better
        assert better_node < worse_node
        assert not worse_node < better_node

    def test_best_bound_maximization(self):
        """In Best-Bound mode for maximization, higher LP objective has priority."""
        Node.switch_to_bb = True
        Node.is_maximization = True

        better_node = Node(node_id=1, parent_id=0, lp_objective=100.0, depth=2)
        worse_node = Node(node_id=2, parent_id=0, lp_objective=50.0, depth=5)

        # For maximization, higher objective is better
        assert better_node < worse_node
        assert not worse_node < better_node

    def test_best_bound_same_objective(self):
        """Nodes with the same LP objective should not have priority over each other."""
        Node.switch_to_bb = True
        Node.is_maximization = False

        node1 = Node(node_id=1, parent_id=0, lp_objective=75.0, depth=2)
        node2 = Node(node_id=2, parent_id=0, lp_objective=75.0, depth=5)

        # Same objective means neither has priority
        assert not node1 < node2
        assert not node2 < node1


class TestNodeDataclass:
    """Tests for the Node dataclass attributes and defaults."""

    def test_node_creation_with_defaults(self):
        """Test that a Node can be created with default values."""
        node = Node(node_id=1, parent_id=None, lp_objective=None)

        assert node.node_id == 1
        assert node.parent_id is None
        assert node.lp_objective is None
        assert node.depth == 0
        assert node.local_constraints == []
        assert node.status == 'PENDING'
        assert node.lp_solution is None
        assert node.vbasis is None
        assert node.cbasis is None

    def test_node_creation_with_all_fields(self):
        """Test that a Node can be created with all fields specified."""
        constraints = [('x1', '<=', 5.0), ('x2', '>=', 2.0)]
        solution = {'x1': 3.0, 'x2': 4.0}
        vbasis = {'x1': 0, 'x2': 1}
        cbasis = {'c1': 0, 'c2': 1}

        node = Node(
            node_id=42,
            parent_id=10,
            lp_objective=123.45,
            depth=7,
            local_constraints=constraints,
            status='SOLVED',
            lp_solution=solution,
            vbasis=vbasis,
            cbasis=cbasis
        )

        assert node.node_id == 42
        assert node.parent_id == 10
        assert node.lp_objective == 123.45
        assert node.depth == 7
        assert node.local_constraints == constraints
        assert node.status == 'SOLVED'
        assert node.lp_solution == solution
        assert node.vbasis == vbasis
        assert node.cbasis == cbasis


class TestNodeInHeap:
    """Tests for using nodes in a heap (priority queue)."""

    def setup_method(self):
        """Reset Node class variables before each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def teardown_method(self):
        """Reset Node class variables after each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def test_heap_ordering_dfs(self):
        """Test that nodes are ordered correctly in a heap for DFS."""
        import heapq

        nodes = [
            Node(node_id=1, parent_id=0, lp_objective=100.0, depth=1),
            Node(node_id=2, parent_id=0, lp_objective=100.0, depth=3),
            Node(node_id=3, parent_id=0, lp_objective=100.0, depth=2),
        ]

        heap = []
        for node in nodes:
            heapq.heappush(heap, node)

        # DFS should pop deepest node first
        popped = heapq.heappop(heap)
        assert popped.depth == 3

    def test_heap_ordering_best_bound_min(self):
        """Test that nodes are ordered correctly in a heap for Best-Bound (min)."""
        import heapq

        Node.switch_to_bb = True
        Node.is_maximization = False

        nodes = [
            Node(node_id=1, parent_id=0, lp_objective=100.0, depth=1),
            Node(node_id=2, parent_id=0, lp_objective=50.0, depth=3),
            Node(node_id=3, parent_id=0, lp_objective=75.0, depth=2),
        ]

        heap = []
        for node in nodes:
            heapq.heappush(heap, node)

        # Best-bound (min) should pop lowest objective first
        popped = heapq.heappop(heap)
        assert popped.lp_objective == 50.0

    def test_heap_ordering_best_bound_max(self):
        """Test that nodes are ordered correctly in a heap for Best-Bound (max)."""
        import heapq

        Node.switch_to_bb = True
        Node.is_maximization = True

        nodes = [
            Node(node_id=1, parent_id=0, lp_objective=100.0, depth=1),
            Node(node_id=2, parent_id=0, lp_objective=50.0, depth=3),
            Node(node_id=3, parent_id=0, lp_objective=75.0, depth=2),
        ]

        heap = []
        for node in nodes:
            heapq.heappush(heap, node)

        # Best-bound (max) should pop highest objective first
        popped = heapq.heappop(heap)
        assert popped.lp_objective == 100.0
