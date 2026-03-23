#!/usr/bin/env python3
"""
Dashboard Generator
===================

Standalone script to generate dashboards from saved optimization results.
This allows iterating on visualizations without re-running the expensive optimization.

Usage:
------
# Generate dashboard from most recent optimization results
python generate_dashboard.py

# Generate from specific results file  
python generate_dashboard.py --file results/optimization_2024-01-01_to_2024-01-07_20241214_120000.pkl

# Generate scenario comparison dashboards
python generate_dashboard.py --scenarios

# List available results
python generate_dashboard.py --list

"""

import argparse
from pathlib import Path
import sys

from lib.paths import OUTPUTS_DIR


def list_available_results():
    """List all saved results files."""
    from lib.results_io import list_saved_results
    
    print("\n" + "="*70)
    print("AVAILABLE SAVED RESULTS")
    print("="*70 + "\n")
    
    df = list_saved_results()
    
    if len(df) == 0:
        print("No saved results found in ./results/")
        print("\nRun one of these first:")
        print("  python model_multi_battery.py   (single optimization)")
        print("  python scenario_analysis.py     (scenario analysis)")
        return
    
    for i, row in df.iterrows():
        print(f"📁 {row['filename']}")
        print(f"   Type: {row['type']}")
        print(f"   Saved: {row['saved_at']}")
        print(f"   Size: {row['size_mb']:.2f} MB")
        print()


def generate_optimization_dashboard(filepath: str = None, output_dir: Path = None):
    """Generate dashboard from single optimization results."""
    from lib.results_io import load_optimization_results, get_latest_results
    from lib.dashboard_multi_battery import (
        create_multi_battery_dashboard,
        create_detailed_site_dashboard,
        print_financial_summary,
        generate_comparison_report
    )
    
    # Get filepath
    if filepath is None:
        filepath = get_latest_results('single_optimization')
        if filepath is None:
            print("❌ No optimization results found. Run model_multi_battery.py first.")
            return False
    
    # Load results
    print("\n" + "="*70)
    print("LOADING RESULTS")
    print("="*70)
    
    results, data, site_configs, metadata = load_optimization_results(filepath)
    
    df = results['df']
    financials = results['financials']
    C_peak = metadata.get('C_peak', None) if metadata else None
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    
    # Generate dashboards
    print("\n" + "="*70)
    print("GENERATING DASHBOARDS")
    print("="*70)
    
    # Print financial summary
    print_financial_summary(financials)
    
    # Main dashboard
    print("\nCreating portfolio dashboard...")
    create_multi_battery_dashboard(
        df, financials, site_configs, data,
        output_file=output_dir / "multi_battery_dashboard.html",
        C_peak=C_peak,
    )
    
    # Comparison report
    print("\nGenerating comparison report...")
    generate_comparison_report(
        df, financials,
        output_file=output_dir / "site_comparison_report.csv"
    )
    
    # Detailed site dashboard (first site)
    if len(site_configs) > 0:
        first_site = site_configs[0].site_id
        print(f"\nCreating detailed dashboard for {first_site}...")
        create_detailed_site_dashboard(
            df, first_site, financials, site_configs[0],
            output_file=output_dir / f"dashboard_{first_site}.html"
        )
    
    print("\n" + "="*70)
    print("✓ DASHBOARDS GENERATED!")
    print("="*70)
    print("\nFiles created in outputs/:")
    print("  📊 multi_battery_dashboard.html")
    print("  📄 site_comparison_report.csv")
    if len(site_configs) > 0:
        print(f"  📈 dashboard_{site_configs[0].site_id}.html")
    
    return True


def generate_scenario_dashboards(filepath: str = None, output_dir: Path = None):
    """Generate dashboards from scenario analysis results."""
    from lib.results_io import load_scenario_results, get_latest_results
    # Ensure pickle can resolve ScenarioResult when scenario_analysis was previously
    # saved while running as `__main__`. The unpickler looks for `__main__.ScenarioResult`.
    import scenario_analysis as _scenario_analysis
    globals()['ScenarioResult'] = _scenario_analysis.ScenarioResult
    from scenario_analysis import (
        create_scenario_overview_dashboard,
        create_master_navigation,
        print_scenario_summary
    )
    from lib.paths import RESULTS_DIR
    
    # Get filepath
    if filepath is None:
        filepath = get_latest_results('scenario_analysis')
        # Fallback: some older pickles may have `type="unknown"` in the metadata.
        # In that case, pick the newest `scenarios_*.pkl` directly by mtime.
        if filepath is None:
            scenario_pickles = sorted(
                list(RESULTS_DIR.glob('scenarios_*.pkl')),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if len(scenario_pickles) > 0:
                filepath = str(scenario_pickles[0])
            else:
                print("❌ No scenario results found. Run scenario_analysis.py first.")
                return False
    
    # Load results
    print("\n" + "="*70)
    print("LOADING SCENARIO RESULTS")
    print("="*70)
    
    results_df, all_results, base_data_info, metadata = load_scenario_results(filepath)
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "scenario_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print summary
    print_scenario_summary(results_df)
    
    # Check for feasible scenarios
    n_feasible = results_df['feasible'].sum() if 'feasible' in results_df.columns else 0
    
    if n_feasible == 0:
        print("\n⚠️  No feasible scenarios to visualize!")
        return False
    
    # Generate dashboards
    print("\n" + "="*70)
    print("GENERATING SCENARIO DASHBOARDS")
    print("="*70)
    
    print("\nCreating merged scenario overview dashboard...")
    create_scenario_overview_dashboard(
        results_df,
        output_file=OUTPUTS_DIR / "scenario_overview_dashboard.html",
        scenario_data=all_results,
    )
    
    print("\nCreating master navigation...")
    create_master_navigation(
        results_df,
        output_dir=output_dir,
        output_file=OUTPUTS_DIR / "scenario_master.html"
    )
    
    print("\n" + "="*70)
    print("✓ SCENARIO DASHBOARDS GENERATED!")
    print("="*70)
    print("\nFiles created in outputs/:")
    print("  🌐 scenario_master.html - Start here!")
    print("  📊 scenario_overview_dashboard.html")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate dashboards from saved optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dashboard.py                     # Latest optimization results
  python generate_dashboard.py --scenarios         # Latest scenario results
  python generate_dashboard.py --list              # List available results
  python generate_dashboard.py --file results/my_results.pkl
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to specific results file (.pkl)'
    )
    
    parser.add_argument(
        '--scenarios', '-s',
        action='store_true',
        help='Generate scenario analysis dashboards (default: single optimization)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available saved results'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for dashboards'
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available_results()
        return 0
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Generate dashboards
    if args.scenarios:
        success = generate_scenario_dashboards(args.file, output_dir)
    else:
        success = generate_optimization_dashboard(args.file, output_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

