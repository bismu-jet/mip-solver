# MIP Solver - Branch and Bound Implementation

A custom Mixed-Integer Programming (MIP) solver implementing the Branch-and-Bound algorithm from scratch, built on top of Gurobi's LP solver.

## Project Overview

This project demonstrates a comprehensive understanding of optimization algorithms by implementing a complete Branch-and-Bound MIP solver with:

- **Cutting Planes**: Gomory Mixed-Integer (GMI) cuts, Knapsack Cover cuts, and Clique cuts
- **Primal Heuristics**: Diving, Feasibility Pump, Coefficient Diving, and RINS
- **Presolve Techniques**: Variable fixing, bound propagation, probing, coefficient tightening
- **Smart Branching**: Pseudocost-based variable selection with reliability branching
- **Hybrid Node Selection**: Depth-first search initially, then best-bound after finding incumbent

## Architecture

```
MIP_SOLVER/
├── main.py                 # CLI entry point
├── config.yaml             # Solver configuration
├── solver/
│   ├── tree_manager.py     # Core B&B algorithm (~470 lines)
│   ├── presolve.py         # Presolve techniques (~500 lines)
│   ├── cuts.py             # Cutting plane generators (~350 lines)
│   ├── heuristics.py       # Primal & improvement heuristics (~330 lines)
│   ├── gurobi_interface.py # LP relaxation solving (~200 lines)
│   ├── problem.py          # MIP problem wrapper
│   ├── node.py             # B&B tree node dataclass
│   └── utilities.py        # Logging setup
└── data/                   # Test MPS instances
```

## Key Design Decisions

### Why wrap Gurobi instead of using it directly?
This project intentionally disables Gurobi's built-in presolve, cuts, and heuristics to demonstrate a custom implementation of these techniques. The educational value lies in implementing these algorithms from scratch while leveraging Gurobi only for solving LP relaxations.

### Node Selection Strategy
The solver uses a hybrid approach:
1. **Initial phase (DFS)**: Explores deep nodes first to quickly find a feasible solution
2. **After first incumbent (Best-Bound)**: Switches to exploring nodes with the best LP bound to prove optimality

This is controlled by static variables in the `Node` class that trigger a heap reorganization when the first incumbent is found.

### Pseudocost Branching
Variables are scored based on historical objective degradation when branched upon. New variables without history fall back to global averages (reliability branching).

### Cut Generation Pipeline
1. **Knapsack Cover Cuts**: For binary knapsack constraints, finds minimal covers violated by LP solution
2. **Clique Cuts**: Builds conflict graph from `x_i + x_j <= 1` constraints, finds maximal cliques
3. **GMI Cuts**: Extracts simplex tableau row for fractional basic variables, applies GMI formula

## Usage

```bash
python main.py --problem_file data/instance.mps --config_file config.yaml
```

## Configuration (config.yaml)

```yaml
solver_params:
  optimality_gap: 0.0001      # 0.01% gap tolerance
  time_limit_seconds: 900     # 15-minute limit
  heuristic_frequency: 20     # Run heuristics every N nodes
  rins_time_limit: 5.0        # Sub-MIP time limit for RINS

strategy:
  node_selection: 'hybrid'    # 'dfs', 'best_bound', or 'hybrid'
  branching_variable: 'pseudocost'  # or 'most_fractional'

presolve_params:
  probing_variable_limit: 100 # Limit probing to top N variables
```

## Dependencies

- `gurobipy` - Gurobi optimizer (requires license)
- `numpy` - Linear algebra operations
- `scipy` - Sparse matrix operations for GMI cuts
- `pyyaml` - Configuration parsing

## Code Quality Notes

- Type hints throughout for clarity
- Comprehensive docstrings explaining algorithm logic
- Proper Gurobi resource management with `dispose()` calls
- Modular design with clear separation of concerns

## Potential Interview Discussion Points

1. **Time Complexity**: GMI cut generation involves basis matrix inversion (O(m^2) to O(m^3))
2. **Memory Management**: Gurobi models must be explicitly disposed to prevent memory leaks
3. **Trade-offs**: Pseudocost branching vs strong branching (speed vs quality)
4. **LP Warm Starting**: Basis information is stored but warm-start implementation is a future enhancement

## Future Enhancements

- Strong branching for improved variable selection
- Warm-start LP solves using stored basis information
- Parallel node processing
- Cut pool management with aging
- More presolve techniques (substitution, clique merging)
