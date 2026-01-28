"""
Integration tests for the MIP Solver.

These tests run the full solver on small MPS problem instances to verify
end-to-end functionality. They require a valid Gurobi license.
"""
import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gurobi_available():
    """Check if Gurobi is available and licensed."""
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        env.dispose()
        return True
    except Exception:
        return False


# Skip all tests in this module if Gurobi is not available
pytestmark = pytest.mark.skipif(
    not gurobi_available(),
    reason="Gurobi not available or not licensed"
)


class TestMIPProblemLoading:
    """Tests for loading MIP problems from files."""

    def test_load_mps_file(self, data_dir, config_path):
        """Test that an MPS file can be loaded successfully."""
        from solver.problem import MIPProblem

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        assert problem.model is not None
        assert len(problem.integer_variable_names) > 0

        problem.dispose()

    def test_integer_variables_identified(self, data_dir):
        """Test that integer variables are correctly identified."""
        from solver.problem import MIPProblem

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        # mas76 should have integer variables
        assert len(problem.integer_variable_names) > 0

        problem.dispose()

    def test_model_parameters_set(self, data_dir):
        """Test that model parameters are configured correctly."""
        from solver.problem import MIPProblem
        from gurobipy import GRB

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        # Check that Gurobi's built-in features are disabled
        assert problem.model.getParamInfo('Presolve')[2] == 0
        assert problem.model.getParamInfo('Cuts')[2] == 0
        assert problem.model.getParamInfo('Heuristics')[2] == 0

        problem.dispose()


class TestLPRelaxation:
    """Tests for solving LP relaxations."""

    def test_solve_root_lp(self, data_dir):
        """Test solving the root LP relaxation."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)
        result = solve_lp_relaxation(problem, local_constraints=[])

        assert result['status'] == 'OPTIMAL'
        assert 'objective' in result
        assert 'solution' in result
        assert isinstance(result['solution'], dict)

        problem.dispose()

    def test_lp_with_branching_constraints(self, data_dir):
        """Test solving LP with branching constraints."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        # Add a simple branching constraint
        if problem.integer_variable_names:
            var_name = problem.integer_variable_names[0]
            constraints = [(var_name, '<=', 0.0)]
            result = solve_lp_relaxation(problem, local_constraints=constraints)

            # Should still be feasible (or infeasible, both are valid outcomes)
            assert result['status'] in ['OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD']

        problem.dispose()


class TestPresolve:
    """Tests for presolve functionality."""

    def test_presolve_runs_without_error(self, data_dir):
        """Test that presolve completes without errors."""
        from solver.problem import MIPProblem
        from solver.presolve import presolve

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)
        config = {'probing_variable_limit': 10}  # Limit probing for speed

        # Should not raise any exceptions
        presolve(problem, config)

        problem.dispose()

    def test_bound_propagation(self, data_dir):
        """Test that bound propagation works."""
        from solver.problem import MIPProblem
        from solver.presolve import propagate_bounds

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        # Should return number of tightenings or -1 for infeasibility
        result = propagate_bounds(problem)

        assert isinstance(result, int)
        assert result >= -1  # -1 means infeasible, >= 0 means valid

        problem.dispose()


class TestCutGeneration:
    """Tests for cutting plane generation."""

    def test_generate_cuts_from_lp_solution(self, data_dir):
        """Test that cut generation runs without errors."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation
        from solver.cuts import generate_all_cuts

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)
        lp_result = solve_lp_relaxation(problem, local_constraints=[])

        if lp_result['status'] == 'OPTIMAL':
            cuts = generate_all_cuts(problem, lp_result)

            # Should return a list (possibly empty)
            assert isinstance(cuts, list)

        problem.dispose()


class TestHeuristics:
    """Tests for heuristic solution finding."""

    def test_initial_heuristic(self, data_dir):
        """Test that the initial heuristic runs without errors."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation
        from solver.heuristics import find_initial_solution

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)
        lp_result = solve_lp_relaxation(problem, local_constraints=[])

        if lp_result['status'] == 'OPTIMAL':
            # Should run without errors (may return None if no solution found)
            result = find_initial_solution(problem, lp_result['solution'], [])

            # Result should be None or a dict
            assert result is None or isinstance(result, dict)

        problem.dispose()


class TestTreeManagerIntegration:
    """Integration tests for the full TreeManager solver."""

    @pytest.mark.timeout(60)  # 60 second timeout
    def test_solver_initialization(self, data_dir, config_path):
        """Test that TreeManager initializes correctly."""
        from solver.tree_manager import TreeManager

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        tm = TreeManager(mps_path, config_path)

        assert tm.problem is not None
        assert tm.config is not None
        assert len(tm.active_nodes) == 0  # Not started yet
        assert tm.incumbent_solution is None
        assert tm.incumbent_objective is None

        tm.problem.dispose()

    @pytest.mark.timeout(120)  # 2 minute timeout for solving
    def test_solver_short_run(self, data_dir, config_path):
        """Test running the solver for a short time."""
        import yaml
        import tempfile
        from solver.tree_manager import TreeManager

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        # Create a modified config with short time limit
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['solver_params']['time_limit_seconds'] = 10  # 10 second limit

        # Write temp config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_config_path = tmp.name

        try:
            tm = TreeManager(mps_path, tmp_config_path)
            solution, objective = tm.solve()

            # Solver should complete (may or may not find solution in 10s)
            # Just verify it doesn't crash
            assert True

            tm.problem.dispose()
        finally:
            os.unlink(tmp_config_path)


class TestNodeOperations:
    """Tests for Node operations within the solver context."""

    def test_node_heap_operations(self):
        """Test that nodes work correctly in a heap."""
        import heapq
        from solver.node import Node

        # Reset node state
        Node.switch_to_bb = False
        Node.is_maximization = False

        nodes = []
        for i in range(5):
            node = Node(
                node_id=i,
                parent_id=None if i == 0 else 0,
                lp_objective=100.0 - i * 10,
                depth=i
            )
            heapq.heappush(nodes, node)

        # In DFS mode, deepest node (depth=4) should pop first
        first = heapq.heappop(nodes)
        assert first.depth == 4

        # Reset
        Node.switch_to_bb = False
        Node.is_maximization = False

    def test_node_switching_strategy(self):
        """Test switching from DFS to Best-Bound."""
        import heapq
        from solver.node import Node

        # Start in DFS mode
        Node.switch_to_bb = False
        Node.is_maximization = False

        # Create nodes with varying depths and objectives
        heap = []
        heapq.heappush(heap, Node(node_id=1, parent_id=0, lp_objective=100.0, depth=1))
        heapq.heappush(heap, Node(node_id=2, parent_id=0, lp_objective=50.0, depth=3))
        heapq.heappush(heap, Node(node_id=3, parent_id=0, lp_objective=75.0, depth=2))

        # DFS: deepest first
        first_dfs = heapq.heappop(heap)
        assert first_dfs.depth == 3

        # Switch to Best-Bound
        Node.switch_to_bb = True
        heapq.heapify(heap)

        # Best-Bound (min): lowest objective first
        first_bb = heapq.heappop(heap)
        assert first_bb.lp_objective == 75.0  # The remaining lowest

        # Reset
        Node.switch_to_bb = False
        Node.is_maximization = False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_constraints_list(self, data_dir):
        """Test solving with an empty constraints list."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)
        result = solve_lp_relaxation(problem, local_constraints=[])

        assert result['status'] in ['OPTIMAL', 'INFEASIBLE', 'UNBOUNDED', 'INF_OR_UNBD']

        problem.dispose()

    def test_conflicting_constraints(self, data_dir):
        """Test that conflicting constraints are handled gracefully."""
        from solver.problem import MIPProblem
        from solver.gurobi_interface import solve_lp_relaxation

        mps_path = os.path.join(data_dir, 'mas76.mps')
        if not os.path.exists(mps_path):
            pytest.skip("Test MPS file not found")

        problem = MIPProblem(mps_path)

        if problem.integer_variable_names:
            var_name = problem.integer_variable_names[0]
            # Conflicting constraints: x >= 10 and x <= 0
            constraints = [
                (var_name, '>=', 10.0),
                (var_name, '<=', 0.0),
            ]
            result = solve_lp_relaxation(problem, local_constraints=constraints)

            # Should be infeasible
            assert result['status'] in ['INFEASIBLE', 'INF_OR_UNBD']

        problem.dispose()
