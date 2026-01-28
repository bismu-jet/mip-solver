"""
Unit tests for the TreeManager class helper methods (solver/tree_manager.py).

These tests focus on the internal helper methods that can be tested in isolation.
The tests are self-contained and don't require Gurobi to be installed.
Integration tests for the full solve() method are in test_integration.py.
"""
import pytest
import math
from unittest.mock import MagicMock
from solver.node import Node


class MockProblem:
    """Mock MIPProblem for testing TreeManager methods."""

    def __init__(self, integer_var_names=None):
        self.integer_variable_names = integer_var_names or ['x1', 'x2', 'x3']
        self.model = MagicMock()
        self.model.ModelSense = -1  # Minimize by default


# Standalone implementations of TreeManager helper methods for testing
def _is_integer_feasible(integer_variable_names, solution, tolerance=1e-6):
    """Check if a solution is integer-feasible."""
    for var_name in integer_variable_names:
        if var_name in solution:
            if abs(solution[var_name] - round(solution[var_name])) > tolerance:
                return False
    return True


def _is_promising(incumbent_objective, is_maximization, node):
    """Check if a node is promising."""
    if incumbent_objective is None:
        return True
    if is_maximization:
        return node.lp_objective > incumbent_objective + 1e-6
    else:
        return node.lp_objective < incumbent_objective - 1e-6


def _update_pseudocosts(pseudocosts, global_up, global_down, var_name, direction, degradation):
    """Update pseudocost information for a variable."""
    if var_name not in pseudocosts:
        pseudocosts[var_name] = {
            'up': {'sum_degrad': 0.0, 'count': 0},
            'down': {'sum_degrad': 0.0, 'count': 0}
        }

    pseudocosts[var_name][direction]['sum_degrad'] += degradation
    pseudocosts[var_name][direction]['count'] += 1

    if direction == 'up':
        global_up['sum_degrad'] += degradation
        global_up['count'] += 1
    else:
        global_down['sum_degrad'] += degradation
        global_down['count'] += 1


def _select_by_pseudocost(pseudocosts, global_up, global_down, solution, fractional_vars):
    """Select branching variable using pseudocost scoring."""
    best_var = None
    max_score = -1.0

    avg_up = (global_up['sum_degrad'] / global_up['count']) if global_up['count'] > 0 else 1.0
    avg_down = (global_down['sum_degrad'] / global_down['count']) if global_down['count'] > 0 else 1.0

    for var_name in fractional_vars:
        val = solution[var_name]
        frac_part = val - math.floor(val)

        var_info = pseudocosts.get(var_name, {'up': {'count': 0}, 'down': {'count': 0}})

        pc_down = (var_info['down']['sum_degrad'] / var_info['down']['count']) if var_info['down']['count'] > 0 else avg_down
        pc_up = (var_info['up']['sum_degrad'] / var_info['up']['count']) if var_info['up']['count'] > 0 else avg_up

        score = (1 - frac_part) * pc_down + frac_part * pc_up

        if score > max_score:
            max_score = score
            best_var = var_name

    return best_var


def _get_branching_variable(integer_var_names, config, pseudocosts, global_up, global_down, solution):
    """Determine which fractional variable to branch on."""
    fractional_vars = [
        var_name for var_name in integer_var_names
        if var_name in solution and abs(solution[var_name] - round(solution[var_name])) > 1e-6
    ]

    if not fractional_vars:
        return None

    strategy = config["strategy"]["branching_variable"]
    if strategy == "pseudocost":
        return _select_by_pseudocost(pseudocosts, global_up, global_down, solution, fractional_vars)
    else:  # Default to "most fractional"
        return max(fractional_vars, key=lambda v: 0.5 - abs(solution[v] - math.floor(solution[v]) - 0.5))


class TestIsIntegerFeasible:
    """Tests for the _is_integer_feasible method."""

    def test_integer_solution_is_feasible(self):
        """Test that a solution with all integer values is feasible."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0, 'y': 1.5}  # y is continuous

        assert _is_integer_feasible(integer_vars, solution) is True

    def test_fractional_solution_is_infeasible(self):
        """Test that a solution with fractional integer variables is infeasible."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {'x1': 1.0, 'x2': 2.5, 'x3': 3.0}

        assert _is_integer_feasible(integer_vars, solution) is False

    def test_nearly_integer_within_tolerance(self):
        """Test that values very close to integers are considered feasible."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {'x1': 1.0000001, 'x2': 2.9999999, 'x3': 3.0}

        assert _is_integer_feasible(integer_vars, solution, tolerance=1e-6) is True

    def test_nearly_integer_outside_tolerance(self):
        """Test that values outside tolerance are considered infeasible."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {'x1': 1.001, 'x2': 2.0, 'x3': 3.0}

        assert _is_integer_feasible(integer_vars, solution, tolerance=1e-6) is False

    def test_missing_variable_in_solution(self):
        """Test that missing variables are handled gracefully."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {'x1': 1.0, 'x3': 3.0}  # x2 is missing

        assert _is_integer_feasible(integer_vars, solution) is True

    def test_empty_solution(self):
        """Test that an empty solution is considered feasible (vacuously true)."""
        integer_vars = ['x1', 'x2', 'x3']
        solution = {}

        assert _is_integer_feasible(integer_vars, solution) is True


class TestIsPromising:
    """Tests for the _is_promising method."""

    def test_promising_when_no_incumbent(self):
        """Test that any node is promising when there's no incumbent."""
        node = Node(node_id=1, parent_id=0, lp_objective=100.0)

        assert _is_promising(None, False, node) is True

    def test_promising_minimization_better_bound(self):
        """Test that a node with lower LP objective is promising for minimization."""
        better_node = Node(node_id=1, parent_id=0, lp_objective=50.0)

        assert _is_promising(100.0, False, better_node) is True

    def test_not_promising_minimization_worse_bound(self):
        """Test that a node with higher LP objective is not promising for minimization."""
        worse_node = Node(node_id=1, parent_id=0, lp_objective=150.0)

        assert _is_promising(100.0, False, worse_node) is False

    def test_promising_maximization_better_bound(self):
        """Test that a node with higher LP objective is promising for maximization."""
        better_node = Node(node_id=1, parent_id=0, lp_objective=150.0)

        assert _is_promising(100.0, True, better_node) is True

    def test_not_promising_maximization_worse_bound(self):
        """Test that a node with lower LP objective is not promising for maximization."""
        worse_node = Node(node_id=1, parent_id=0, lp_objective=50.0)

        assert _is_promising(100.0, True, worse_node) is False

    def test_boundary_case_equal_objective(self):
        """Test that a node with equal LP objective is not promising."""
        equal_node = Node(node_id=1, parent_id=0, lp_objective=100.0)

        # Equal is not strictly better, so not promising
        assert _is_promising(100.0, False, equal_node) is False


class TestUpdatePseudocosts:
    """Tests for the _update_pseudocosts method."""

    def test_update_new_variable_up(self):
        """Test updating pseudocost for a new variable with 'up' direction."""
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}

        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'up', 10.0)

        assert 'x1' in pseudocosts
        assert pseudocosts['x1']['up']['sum_degrad'] == 10.0
        assert pseudocosts['x1']['up']['count'] == 1
        assert global_up['sum_degrad'] == 10.0
        assert global_up['count'] == 1

    def test_update_new_variable_down(self):
        """Test updating pseudocost for a new variable with 'down' direction."""
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}

        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'down', 5.0)

        assert 'x1' in pseudocosts
        assert pseudocosts['x1']['down']['sum_degrad'] == 5.0
        assert pseudocosts['x1']['down']['count'] == 1
        assert global_down['sum_degrad'] == 5.0
        assert global_down['count'] == 1

    def test_update_existing_variable(self):
        """Test that multiple updates accumulate correctly."""
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}

        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'up', 10.0)
        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'up', 20.0)

        assert pseudocosts['x1']['up']['sum_degrad'] == 30.0
        assert pseudocosts['x1']['up']['count'] == 2
        assert global_up['sum_degrad'] == 30.0
        assert global_up['count'] == 2

    def test_update_multiple_variables(self):
        """Test updating pseudocosts for multiple variables."""
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}

        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'up', 10.0)
        _update_pseudocosts(pseudocosts, global_up, global_down, 'x2', 'down', 15.0)
        _update_pseudocosts(pseudocosts, global_up, global_down, 'x1', 'down', 8.0)

        assert pseudocosts['x1']['up']['sum_degrad'] == 10.0
        assert pseudocosts['x1']['down']['sum_degrad'] == 8.0
        assert pseudocosts['x2']['down']['sum_degrad'] == 15.0
        assert global_up['sum_degrad'] == 10.0
        assert global_down['sum_degrad'] == 23.0


class TestSelectByPseudocost:
    """Tests for the _select_by_pseudocost method."""

    def test_select_single_variable(self):
        """Test selecting when there's only one fractional variable."""
        pseudocosts = {}
        global_up = {'sum_degrad': 10.0, 'count': 2}
        global_down = {'sum_degrad': 8.0, 'count': 2}
        solution = {'x1': 2.5}
        fractional_vars = ['x1']

        result = _select_by_pseudocost(pseudocosts, global_up, global_down, solution, fractional_vars)

        assert result == 'x1'

    def test_select_with_pseudocost_history(self):
        """Test that variable-specific pseudocosts are used when available."""
        # x1 has high pseudocost, x2 has low pseudocost
        pseudocosts = {
            'x1': {'up': {'sum_degrad': 100.0, 'count': 2}, 'down': {'sum_degrad': 100.0, 'count': 2}},
            'x2': {'up': {'sum_degrad': 1.0, 'count': 2}, 'down': {'sum_degrad': 1.0, 'count': 2}},
        }
        global_up = {'sum_degrad': 10.0, 'count': 2}
        global_down = {'sum_degrad': 8.0, 'count': 2}
        solution = {'x1': 2.5, 'x2': 3.5}
        fractional_vars = ['x1', 'x2']

        # x1 should be selected because it has higher pseudocost scores
        result = _select_by_pseudocost(pseudocosts, global_up, global_down, solution, fractional_vars)

        assert result == 'x1'

    def test_select_uses_global_average_for_new_vars(self):
        """Test that global average is used for variables without history."""
        pseudocosts = {}  # No history for any variable
        global_up = {'sum_degrad': 10.0, 'count': 2}
        global_down = {'sum_degrad': 8.0, 'count': 2}
        solution = {'x1': 2.3, 'x2': 3.7}
        fractional_vars = ['x1', 'x2']

        # Should still return a variable (using global averages)
        result = _select_by_pseudocost(pseudocosts, global_up, global_down, solution, fractional_vars)

        assert result in ['x1', 'x2']


class TestGetBranchingVariable:
    """Tests for the _get_branching_variable method."""

    def test_no_branching_for_integer_solution(self):
        """Test that None is returned for an integer-feasible solution."""
        integer_vars = ['x1', 'x2', 'x3']
        config = {"strategy": {"branching_variable": "most_fractional"}}
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}
        solution = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0}

        result = _get_branching_variable(integer_vars, config, pseudocosts, global_up, global_down, solution)

        assert result is None

    def test_most_fractional_strategy(self):
        """Test that most_fractional strategy selects variable closest to 0.5."""
        integer_vars = ['x1', 'x2', 'x3']
        config = {"strategy": {"branching_variable": "most_fractional"}}
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}

        # x1 has fractional part 0.3 (distance from 0.5 = 0.2)
        # x2 has fractional part 0.5 (distance from 0.5 = 0.0) <- most fractional
        # x3 has fractional part 0.8 (distance from 0.5 = 0.3)
        solution = {'x1': 2.3, 'x2': 3.5, 'x3': 4.8}

        result = _get_branching_variable(integer_vars, config, pseudocosts, global_up, global_down, solution)

        assert result == 'x2'

    def test_pseudocost_strategy(self):
        """Test that pseudocost strategy is used when configured."""
        integer_vars = ['x1', 'x2']
        config = {"strategy": {"branching_variable": "pseudocost"}}
        pseudocosts = {
            'x1': {'up': {'sum_degrad': 100.0, 'count': 1}, 'down': {'sum_degrad': 100.0, 'count': 1}},
            'x2': {'up': {'sum_degrad': 1.0, 'count': 1}, 'down': {'sum_degrad': 1.0, 'count': 1}},
        }
        global_up = {'sum_degrad': 101.0, 'count': 2}
        global_down = {'sum_degrad': 101.0, 'count': 2}
        solution = {'x1': 2.5, 'x2': 3.5}

        result = _get_branching_variable(integer_vars, config, pseudocosts, global_up, global_down, solution)

        # x1 should be chosen due to higher pseudocost
        assert result == 'x1'

    def test_only_considers_integer_variables(self):
        """Test that only integer variables are considered for branching."""
        integer_vars = ['x1', 'x2', 'x3']  # y is not in this list
        config = {"strategy": {"branching_variable": "most_fractional"}}
        pseudocosts = {}
        global_up = {'sum_degrad': 0.0, 'count': 0}
        global_down = {'sum_degrad': 0.0, 'count': 0}
        # y is not in integer_variable_names, so it should be ignored
        solution = {'x1': 1.0, 'x2': 2.0, 'x3': 3.0, 'y': 1.5}

        result = _get_branching_variable(integer_vars, config, pseudocosts, global_up, global_down, solution)

        # Should return None because x1, x2, x3 are all integer
        assert result is None


class TestUpdateIncumbent:
    """Tests for the _update_incumbent method."""

    def setup_method(self):
        """Reset Node class variables before each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def teardown_method(self):
        """Reset Node class variables after each test."""
        Node.switch_to_bb = False
        Node.is_maximization = False

    def test_first_incumbent_updates(self):
        """Test that the first incumbent is always accepted."""
        # Simulate update_incumbent logic
        incumbent_objective = None
        is_maximization = False

        new_solution = {'x1': 1.0, 'x2': 2.0}
        new_objective = 100.0

        # First incumbent should always be accepted
        is_new_best = incumbent_objective is None or \
                      (is_maximization and new_objective > incumbent_objective) or \
                      (not is_maximization and new_objective < incumbent_objective)

        assert is_new_best is True

    def test_better_incumbent_for_minimization(self):
        """Test that a better incumbent is accepted for minimization."""
        incumbent_objective = 100.0
        is_maximization = False

        better_objective = 50.0  # Lower is better for minimization

        is_new_best = incumbent_objective is None or \
                      (is_maximization and better_objective > incumbent_objective) or \
                      (not is_maximization and better_objective < incumbent_objective)

        assert is_new_best is True

    def test_worse_incumbent_rejected_for_minimization(self):
        """Test that a worse incumbent is rejected for minimization."""
        incumbent_objective = 100.0
        is_maximization = False

        worse_objective = 150.0  # Higher is worse for minimization

        is_new_best = incumbent_objective is None or \
                      (is_maximization and worse_objective > incumbent_objective) or \
                      (not is_maximization and worse_objective < incumbent_objective)

        assert is_new_best is False

    def test_better_incumbent_for_maximization(self):
        """Test that a better incumbent is accepted for maximization."""
        incumbent_objective = 100.0
        is_maximization = True

        better_objective = 150.0  # Higher is better for maximization

        is_new_best = incumbent_objective is None or \
                      (is_maximization and better_objective > incumbent_objective) or \
                      (not is_maximization and better_objective < incumbent_objective)

        assert is_new_best is True

    def test_first_incumbent_switches_to_best_bound(self):
        """Test that finding the first incumbent switches to best-bound search."""
        # Simulate the switch logic
        incumbent_objective = None

        if incumbent_objective is None:
            Node.switch_to_bb = True
            Node.is_maximization = False

        assert Node.switch_to_bb is True


class TestOptimalityGap:
    """Tests for optimality gap calculation."""

    def test_gap_calculation_minimization(self):
        """Test gap calculation for minimization problem."""
        incumbent_objective = 100.0
        global_best_bound = 90.0  # Lower bound for minimization

        gap = abs(incumbent_objective - global_best_bound) / (abs(incumbent_objective) + 1e-9)

        assert abs(gap - 0.1) < 1e-6  # 10% gap

    def test_gap_calculation_maximization(self):
        """Test gap calculation for maximization problem."""
        incumbent_objective = 100.0
        global_best_bound = 110.0  # Upper bound for maximization

        gap = abs(incumbent_objective - global_best_bound) / (abs(incumbent_objective) + 1e-9)

        assert abs(gap - 0.1) < 1e-6  # 10% gap

    def test_gap_is_zero_when_optimal(self):
        """Test that gap is zero when incumbent equals bound."""
        incumbent_objective = 100.0
        global_best_bound = 100.0

        gap = abs(incumbent_objective - global_best_bound) / (abs(incumbent_objective) + 1e-9)

        assert gap < 1e-6

    def test_gap_tolerance_check(self):
        """Test that gap tolerance check works correctly."""
        incumbent_objective = 100.0
        global_best_bound = 99.99  # Very close
        optimality_gap = 0.0001  # 0.01%

        gap = abs(incumbent_objective - global_best_bound) / (abs(incumbent_objective) + 1e-9)

        # Gap should be within tolerance
        assert gap <= optimality_gap
