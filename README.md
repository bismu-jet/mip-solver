# MIP Solver - Branch and Bound Implementation

[![Tests](https://github.com/bismu-jet/mip-solver/actions/workflows/tests.yml/badge.svg)](https://github.com/bismu-jet/mip-solver/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A custom Mixed-Integer Programming (MIP) solver implementing the Branch-and-Bound algorithm from scratch, built on top of Gurobi's LP solver.

## Project Overview

This project demonstrates a comprehensive understanding of optimization algorithms by implementing a complete Branch-and-Bound MIP solver with:

- **Cutting Planes**: Gomory Mixed-Integer (GMI) cuts, Knapsack Cover cuts, and Clique cuts
- **Primal Heuristics**: Diving, Feasibility Pump, Coefficient Diving, and RINS
- **Presolve Techniques**: Variable fixing, bound propagation, probing, coefficient tightening
- **Smart Branching**: Pseudocost-based variable selection with reliability branching
- **Hybrid Node Selection**: Depth-first search initially, then best-bound after finding incumbent

## Installation

### Prerequisites

- Python 3.9 or higher
- Gurobi Optimizer with a valid license ([free academic licenses available](https://www.gurobi.com/academia/academic-program-and-licenses/))

### Setup

```bash
# Clone the repository
git clone https://github.com/bismu-jet/PI-final.git
cd PI-final

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r MIP_SOLVER/requirements.txt
```

## Usage

```bash
cd MIP_SOLVER
python main.py --problem_file data/mas76.mps --config_file config.yaml
```

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
├── tests/                  # Comprehensive test suite
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

## Configuration

Edit `MIP_SOLVER/config.yaml` to customize solver behavior:

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

## Testing

```bash
# Run all tests
cd MIP_SOLVER
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=solver --cov-report=term-missing
```

## Technical Discussion Points

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
