# LBLDE (Level-Based Learning Differential Evolution) - Python Implementation

Implementation of the algorithm from:
> Qiao, K., Liang, J., Qu, B., Yu, K., Yue, C., & Song, H. (2022). Differential Evolution with Level-Based Learning Mechanism. *Complex System Modeling and Simulation*, 2(1), 35-58.

## 📋 What the Paper Tested

The paper evaluated LBLDE on the **CEC 2017 benchmark suite**:

### Test Functions (30 total)
- **F1-F3**: Unimodal functions
- **F4-F10**: Simple multimodal functions  
- **F11-F20**: Hybrid functions
- **F21-F30**: Composition functions

### Experimental Protocol
- **Dimensions**: 10D, 30D, 50D, 100D
- **Runs**: 51 independent runs per function
- **MaxFES**: 10,000 × D function evaluations
- **Population Size**: 
  - 100 for D ≤ 50
  - 160 for D = 100

## 🚀 Quick Start

### Installation

```bash
# Install CEC 2017 benchmark (choose one):

# Option 1: opfunu (recommended - pure Python)
pip install opfunu

# Option 2: cec2017-py 
pip install cec2017-py

# Option 3: From source
git clone https://github.com/tilleyd/cec2017-py
cd cec2017-py
python setup.py install
```

### Basic Usage

```python
from lblde import LBLDE
import numpy as np

# Define your objective function
def sphere(x):
    return np.sum(x**2)

# Set bounds
D = 10  # dimension
bounds = np.array([[-100, 100]] * D)

# Create optimizer
optimizer = LBLDE(
    objective_func=sphere,
    bounds=bounds,
    NP=100,           # Population size
    NL=4,             # Number of levels
    NLB=1,            # Bottom levels with CR=1
    mu_CR_ini=0.35,   # Initial CR mean
    max_fes=100000    # Max function evaluations
)

# Run optimization
best_solution, best_fitness, history = optimizer.optimize()

print(f"Best fitness: {best_fitness:.6e}")
```

### CEC 2017 Testing

```python
from opfunu.cec_based.cec2017 import F12017
from lblde import LBLDE

# Initialize CEC 2017 function
D = 10
func = F12017(ndim=D)

# Run LBLDE
optimizer = LBLDE(
    objective_func=func.evaluate,
    bounds=func.bounds,
    NP=100,
    max_fes=10000 * D
)

best_sol, best_fit, history = optimizer.optimize()

# Calculate error (as in paper)
error = best_fit - func.f_global
print(f"Error: {error:.6e}")
```

### Multiple Runs (As in Paper)

```python
from lblde import run_multiple_trials

# Run 51 independent trials
results = run_multiple_trials(
    func=func.evaluate,
    bounds=func.bounds,
    D=10,
    n_runs=51,
    max_fes=100000
)

print(f"Mean ± Std: {results['mean']:.6e} ± {results['std']:.6e}")
```

## 📊 Reproducing Paper Results

To replicate the experiments from Tables 5-8:

```python
# Test on all 30 CEC 2017 functions
from opfunu.cec_based.cec2017 import *

dimensions = [10, 30, 50, 100]
functions = [F12017, F22017, ..., F302017]  # All 30 functions

for D in dimensions:
    for FuncClass in functions:
        func = FuncClass(ndim=D)
        results = run_multiple_trials(
            func=func.evaluate,
            bounds=func.bounds,
            D=D,
            n_runs=51,
            max_fes=10000 * D
        )
        # Save results...
```

## 🔧 Algorithm Parameters

### Default Settings (from paper)
```python
NP = 100        # Population size (160 for 100D)
NL = 4          # Number of levels
NLB = 1         # Bottom levels with CR=1
mu_CR_ini = 0.35  # Initial CR mean
mu_F = 0.5      # Initial F mean
c = 0.1         # Learning rate
```

### Key Components

1. **Level-Based Learning**: Population divided into 4 levels
   - Level 1: Top 25% (learns from top 5%)
   - Level 2-3: Middle 50% (learns from more individuals)
   - Level 4: Bottom 25% (CR=1, high exploration)

2. **Adaptive Parameters**:
   - CR: Normal distribution, μ_CR adapts
   - F: Cauchy distribution, μ_F adapts

3. **Mutation Strategy**: DE/current-to-pbest/1

## 📈 Expected Results (10D)

Based on Table 5 in the paper, you should get approximately:

| Function | Mean Error | Your Result |
|----------|------------|-------------|
| F1       | 0.00e+00   | ___         |
| F2       | 0.00e+00   | ___         |
| F3       | 0.00e+00   | ___         |
| F4       | 3.17e-03   | ___         |
| F5       | 2.62e+00   | ___         |
| ...      | ...        | ___         |

## 🔍 Comparison with Other Algorithms

The paper compares LBLDE with:
- EAGDE
- EFADE
- AMECoDEs
- TSDE
- RNDE
- MPEDE
- TVDE

LBLDE showed superior or competitive performance, ranking 2nd overall.

## 📁 File Structure

```
lblde/
├── lblde.py              # Main LBLDE implementation
├── cec2017_testing.py    # CEC 2017 testing utilities
├── README.md             # This file
└── examples/
    ├── basic_usage.py
    ├── cec2017_single.py
    └── full_experiment.py
```

## ⚙️ Advanced Usage

### Custom Stopping Criteria

```python
class CustomLBLDE(LBLDE):
    def optimize(self, target_fitness=1e-8, **kwargs):
        # Add custom stopping condition
        while self.FES < self.max_fes:
            # ... optimization loop ...
            if self.best_fitness < target_fitness:
                break
        return self.best_solution, self.best_fitness
```

### Parallel Evaluation

```python
from multiprocessing import Pool

def parallel_runs(n_runs=51):
    with Pool(processes=8) as pool:
        results = pool.starmap(single_run, [(i,) for i in range(n_runs)])
    return results
```

## 📊 Visualization

```python
import matplotlib.pyplot as plt

# Plot convergence curve
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.yscale('log')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('LBLDE Convergence')
plt.grid(True)
plt.show()
```

## 🐛 Troubleshooting

### Issue: Slow convergence
- Try increasing population size (NP)
- Adjust mu_CR_ini (try 0.2 or 0.5)

### Issue: Premature convergence
- Increase NL (number of levels)
- Decrease mu_CR_ini for more exploration

### Issue: CEC 2017 functions not found
```bash
# Make sure opfunu is installed correctly
pip install --upgrade opfunu

# Or use alternative library
pip install cec2017-py
```

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@article{qiao2022differential,
  title={Differential Evolution with Level-Based Learning Mechanism},
  author={Qiao, Kangjia and Liang, Jing and Qu, Boyang and Yu, Kunjie and Yue, Caitong and Song, Hui},
  journal={Complex System Modeling and Simulation},
  volume={2},
  number={1},
  pages={35--58},
  year={2022},
  doi={10.23919/CSMS.2022.0004}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Parallelization support
- GPU acceleration
- Additional benchmark suites (CEC 2005, 2013, etc.)
- Constrained optimization support

## 📝 License

This implementation is for research and educational purposes.

## 🔗 Resources

- [Paper PDF](https://www.doi.org/10.23919/CSMS.2022.0004)
- [CEC 2017 Benchmark](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)
- [opfunu Documentation](https://opfunu.readthedocs.io/)
- [Original C Implementation](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)

## ❓ FAQ

**Q: Why are my results different from the paper?**  
A: Results may vary due to different random seeds. Run 51 trials and compare mean±std.

**Q: Can I use this for constrained optimization?**  
A: The base algorithm is for unconstrained problems. Modifications needed for constraints.

**Q: What dimensions should I use for my problem?**  
A: Start with the same dimension as your problem. Adjust NP based on complexity.

**Q: How long does a full CEC 2017 test take?**  
A: For 30 functions × 4 dimensions × 51 runs ≈ 6,120 runs. On a modern CPU: 2-8 hours.

---

**Last Updated**: January 2026  
**Implementation**: Python 3.7+  
**Dependencies**: numpy, pandas (optional), matplotlib (optional), opfunu (for CEC 2017)