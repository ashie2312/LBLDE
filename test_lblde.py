"""
Simple LBLDE Test Script
========================
This is a standalone script to test LBLDE on CEC 2017 functions.
Place this file in the same directory as your LBLDE implementation.
"""

import numpy as np
import sys

# Import LBLDE classes
# Make sure lblde.py is in the same directory or adjust the import
try:
    from lblde import LBLDE, run_multiple_trials
except ImportError:
    print("Error: Cannot import LBLDE. Make sure lblde.py is in the same directory.")
    sys.exit(1)


def test_basic_functions():
    """Test LBLDE on basic benchmark functions"""
    print("="*80)
    print("Testing LBLDE on Basic Benchmark Functions")
    print("="*80)
    
    # Test functions
    def sphere(x):
        return np.sum(x**2)
    
    def rastrigin(x):
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    def rosenbrock(x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    # Test configuration
    D = 10
    tests = [
        ("Sphere", sphere, [-100, 100]),
        ("Rastrigin", rastrigin, [-5.12, 5.12]),
        ("Rosenbrock", rosenbrock, [-30, 30])
    ]
    
    for name, func, bound_range in tests:
        print(f"\n{'-'*80}")
        print(f"Function: {name} ({D}D)")
        print(f"{'-'*80}")
        
        bounds = np.array([bound_range] * D)
        
        optimizer = LBLDE(
            objective_func=func,
            bounds=bounds,
            NP=100,
            NL=4,
            NLB=1,
            mu_CR_ini=0.35,
            max_fes=10000 * D,
            seed=42
        )
        
        best_sol, best_fit, history = optimizer.optimize(verbose=False)
        
        print(f"Best Fitness: {best_fit:.6e}")
        print(f"Convergence Rate: {len(history)} generations")
        
    print("\n" + "="*80)
    print("Basic tests completed!")
    print("="*80)


def test_single_cec2017():
    """Test a single CEC 2017 function"""
    print("\n" + "="*80)
    print("Testing LBLDE on CEC 2017 Function")
    print("="*80)
    
    try:
        from opfunu.cec_based import cec2017
        print("✓ CEC 2017 library loaded successfully!")
    except ImportError:
        print("✗ opfunu not installed.")
        print("  Install with: pip install opfunu")
        print("  Skipping CEC 2017 test...")
        return
    
    # Test F1 (Shifted and Rotated Bent Cigar Function)
    D = 10
    print(f"\nTesting F1 (Shifted and Rotated Bent Cigar) - {D}D")
    
    func_obj = cec2017.F12017(ndim=D)
    
    print(f"Bounds: {func_obj.bounds[0]}")
    print(f"Global optimum value: {func_obj.f_global}")
    
    print("\nRunning LBLDE...")
    optimizer = LBLDE(
        objective_func=func_obj.evaluate,
        bounds=func_obj.bounds,
        NP=100,
        NL=4,
        NLB=1,
        mu_CR_ini=0.35,
        max_fes=10000 * D,
        seed=42
    )
    
    best_sol, best_fit, history = optimizer.optimize(verbose=True)
    
    # Calculate error (as in the paper)
    error = best_fit - func_obj.f_global
    
    print(f"\n{'='*80}")
    print("Results:")
    print(f"  Best Fitness: {best_fit:.6e}")
    print(f"  Global Optimum: {func_obj.f_global:.6e}")
    print(f"  Error: {error:.6e}")
    print(f"  Best Solution (first 5): {best_sol[:5]}")
    print(f"{'='*80}")


def test_multiple_runs_cec2017():
    """Run multiple trials on a CEC 2017 function (as in paper)"""
    print("\n" + "="*80)
    print("Multiple Runs Test (51 runs as in paper)")
    print("="*80)
    
    try:
        from opfunu.cec_based import cec2017
    except ImportError:
        print("✗ opfunu not installed. Skipping...")
        return
    
    D = 10
    n_runs = 51  # As in the paper
    
    print(f"\nFunction: F1 ({D}D)")
    print(f"Number of runs: {n_runs}")
    print(f"Max FES: {10000 * D}")
    
    func_obj = cec2017.F12017(ndim=D)
    
    print("\nRunning trials (this may take a few minutes)...")
    
    results = run_multiple_trials(
        func=func_obj.evaluate,
        bounds=func_obj.bounds,
        D=D,
        n_runs=n_runs,
        max_fes=10000 * D,
        verbose=False
    )
    
    # Calculate errors
    errors = results['all_best'] - func_obj.f_global
    
    print(f"\n{'='*80}")
    print("Statistical Results:")
    print(f"  Mean Error: {np.mean(errors):.6e}")
    print(f"  Std Error: {np.std(errors):.6e}")
    print(f"  Median Error: {np.median(errors):.6e}")
    print(f"  Min Error: {np.min(errors):.6e}")
    print(f"  Max Error: {np.max(errors):.6e}")
    print(f"\n  Paper reports (F1, 10D): 0.00e+00 ± 0.00e+00")
    print(f"{'='*80}")


def test_multiple_cec2017_functions():
    """Test multiple CEC 2017 functions"""
    print("\n" + "="*80)
    print("Testing Multiple CEC 2017 Functions")
    print("="*80)
    
    try:
        from opfunu.cec_based import cec2017
    except ImportError:
        print("✗ opfunu not installed. Skipping...")
        return
    
    D = 10
    n_runs = 5  # Reduced for quick test
    
    # Test first 5 functions
    functions = [
        (1, cec2017.F12017, "Shifted and Rotated Bent Cigar"),
        (2, cec2017.F22017, "Shifted and Rotated Sum of Different Power"),
        (3, cec2017.F32017, "Shifted and Rotated Zakharov"),
        (4, cec2017.F42017, "Shifted and Rotated Rosenbrock"),
        (5, cec2017.F52017, "Shifted and Rotated Rastrigin")
    ]
    
    results_list = []
    
    for func_num, FuncClass, func_name in functions:
        print(f"\n{'-'*80}")
        print(f"F{func_num}: {func_name} ({D}D)")
        print(f"{'-'*80}")
        
        func_obj = FuncClass(ndim=D)
        
        # Single run for quick test
        optimizer = LBLDE(
            objective_func=func_obj.evaluate,
            bounds=func_obj.bounds,
            NP=100,
            max_fes=10000 * D,
            seed=42
        )
        
        best_sol, best_fit, history = optimizer.optimize(verbose=False)
        error = best_fit - func_obj.f_global
        
        print(f"  Best Fitness: {best_fit:.6e}")
        print(f"  Error: {error:.6e}")
        
        results_list.append({
            'Function': f'F{func_num}',
            'Name': func_name,
            'Best_Fitness': best_fit,
            'Error': error
        })
    
    print(f"\n{'='*80}")
    print("Summary of Results:")
    print(f"{'='*80}")
    for result in results_list:
        print(f"{result['Function']}: Error = {result['Error']:.6e}")
    print(f"{'='*80}")


def main():
    """Main function with menu"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              LBLDE Testing Script                                    ║
    ║                                                                      ║
    ║  Select a test to run:                                               ║
    ║                                                                      ║
    ║  1. Basic benchmark functions (Sphere, Rastrigin, Rosenbrock)       ║
    ║  2. Single CEC 2017 function (F1)                                    ║
    ║  3. Multiple runs on CEC 2017 F1 (51 runs)                          ║
    ║  4. Test multiple CEC 2017 functions (F1-F5)                        ║
    ║  5. Run all tests                                                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        test_basic_functions()
    elif choice == '2':
        test_single_cec2017()
    elif choice == '3':
        test_multiple_runs_cec2017()
    elif choice == '4':
        test_multiple_cec2017_functions()
    elif choice == '5':
        print("\nRunning all tests...\n")
        test_basic_functions()
        test_single_cec2017()
        test_multiple_runs_cec2017()
        test_multiple_cec2017_functions()
    else:
        print("\nInvalid choice. Running basic tests...")
        test_basic_functions()
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  • To test all 30 CEC 2017 functions, modify test_multiple_cec2017_functions()")
    print("  • To test different dimensions (30D, 50D, 100D), change D variable")
    print("  • To run full experiment (51 runs × 30 functions), use the full script")
    print("="*80)


if __name__ == "__main__":
    main()