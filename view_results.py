#!/usr/bin/env python3
"""
View Saved Scenario Results
============================

Load and visualize the optimization results stored in the results folder.
Handles .pkl.gz compressed files from the scenario runs.

Usage:
    python view_results.py                  # List available results and generate all dashboards
    python view_results.py --list           # Just list available results
    python view_results.py --scenario btm50 # Generate dashboard for specific scenario
"""

import gzip
import pickle
import argparse
from pathlib import Path
import sys

from lib.paths import RESULTS_DIR, OUTPUTS_DIR


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles classes pickled from __main__."""
    
    def find_class(self, module, name):
        # If class was pickled from __main__, look in model_multi_battery instead
        if module == '__main__' and name in ('SiteConfig', 'BatterySpec'):
            import model_multi_battery
            return getattr(model_multi_battery, name)
        return super().find_class(module, name)


def load_scenario_data(filepath: Path) -> dict:
    """Load a .pkl.gz scenario results file."""
    print(f"\n📂 Loading: {filepath.name}")
    
    with gzip.open(filepath, 'rb') as f:
        data = CustomUnpickler(f).load()
    
    print(f"   ✓ Loaded {data['df'].shape[0]:,} timesteps across {len(data['site_configs'])} sites")
    
    return data


def list_available_scenarios():
    """List all available scenario results."""
    print("\n" + "="*70)
    print("AVAILABLE SCENARIO RESULTS")
    print("="*70)
    
    pkl_files = list(RESULTS_DIR.glob("dashboard_data_*.pkl.gz"))
    json_files = list(RESULTS_DIR.glob("summary_*.json"))
    
    if not pkl_files:
        print("\n❌ No results found in ./results/")
        print("   Run model_multi_battery.py first to generate results.")
        return []
    
    scenarios = []
    for pkl_file in sorted(pkl_files):
        # Extract scenario name from filename
        # dashboard_data_btm50_2024-01-01_to_2024-12-31.pkl.gz -> btm50
        name_parts = pkl_file.stem.replace('.pkl', '').split('_')
        scenario_name = name_parts[2] if len(name_parts) > 2 else pkl_file.stem
        
        # Find matching summary
        summary_file = RESULTS_DIR / f"summary_{scenario_name}_{name_parts[3]}_{name_parts[4]}_{name_parts[5]}.json"
        
        size_mb = pkl_file.stat().st_size / 1024 / 1024
        
        print(f"\n📁 {scenario_name.upper()}")
        print(f"   Data file: {pkl_file.name}")
        print(f"   Size: {size_mb:.1f} MB")
        
        # Read summary if exists
        if summary_file.exists():
            import json
            with open(summary_file) as f:
                content = f.read()
            # Parse only the JSON part (ends at first closing brace on its own line)
            try:
                json_end = content.index('\n}') + 2
                summary = json.loads(content[:json_end])
                print(f"   Net Benefit: €{summary.get('net_benefit', 0):,.0f}")
                print(f"   FCR Revenue: €{summary.get('fcr_revenue', 0):,.0f}")
                print(f"   Sites: {summary.get('sites_count', '?')}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"   (Could not parse summary: {e})")
        
        scenarios.append({
            'name': scenario_name,
            'file': pkl_file,
            'summary_file': summary_file if summary_file.exists() else None
        })
    
    print("\n" + "="*70)
    return scenarios


def generate_dashboard_for_scenario(scenario_name: str, output_suffix: str = ""):
    """Generate dashboard for a specific scenario."""
    from lib.dashboard_multi_battery import (
        create_multi_battery_dashboard,
        create_detailed_site_dashboard,
        print_financial_summary,
        generate_comparison_report
    )
    
    # Find the scenario file
    pattern = f"dashboard_data_{scenario_name}_*.pkl.gz"
    files = list(RESULTS_DIR.glob(pattern))
    
    if not files:
        print(f"❌ No results found for scenario '{scenario_name}'")
        print(f"   Looking for: {pattern}")
        return False
    
    filepath = files[0]  # Take the first match
    
    # Load data
    data = load_scenario_data(filepath)
    
    df = data['df']
    financials = data['financials']
    site_configs = data['site_configs']
    input_data = data['data']
    
    # Print financial summary
    print("\n" + "="*70)
    print(f"FINANCIAL SUMMARY - {scenario_name.upper()}")
    print("="*70)
    print_financial_summary(financials)
    
    # Output directory
    output_dir = OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main dashboard
    print("\n📊 Generating portfolio dashboard...")
    output_file = output_dir / f"dashboard_{scenario_name}{output_suffix}.html"
    create_multi_battery_dashboard(
        df, financials, site_configs, input_data,
        output_file=output_file
    )
    
    # Generate comparison report
    print("\n📄 Generating comparison report...")
    report_file = output_dir / f"comparison_{scenario_name}{output_suffix}.csv"
    generate_comparison_report(df, financials, output_file=report_file)
    
    # Generate detailed dashboard for first site
    if len(site_configs) > 0:
        first_site = site_configs[0].site_id
        print(f"\n📈 Generating detailed dashboard for {first_site}...")
        site_file = output_dir / f"dashboard_{scenario_name}_{first_site}{output_suffix}.html"
        create_detailed_site_dashboard(
            df, first_site, financials, site_configs[0],
            output_file=site_file
        )
    
    print("\n" + "="*70)
    print(f"✅ DASHBOARDS GENERATED FOR {scenario_name.upper()}!")
    print("="*70)
    print(f"\n   📊 {output_file.name}")
    print(f"   📄 {report_file.name}")
    if len(site_configs) > 0:
        print(f"   📈 {site_file.name}")
    print(f"\n   Open the HTML files in your browser to view the dashboards.")
    
    return True


def generate_all_dashboards():
    """Generate dashboards for all available scenarios."""
    scenarios = list_available_scenarios()
    
    if not scenarios:
        return
    
    print("\n" + "="*70)
    print("GENERATING DASHBOARDS")
    print("="*70)
    
    for scenario in scenarios:
        print(f"\n{'─'*40}")
        generate_dashboard_for_scenario(scenario['name'])


def create_comparison_overview():
    """Create an overview comparing all scenarios."""
    import json
    
    scenarios = list_available_scenarios()
    if not scenarios:
        return
    
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    
    print(f"\n{'Scenario':<12} {'Net Benefit':>14} {'FCR Revenue':>14} {'Peak Savings':>14} {'BTM Savings':>14}")
    print("─" * 70)
    
    for scenario in scenarios:
        if scenario['summary_file'] and scenario['summary_file'].exists():
            with open(scenario['summary_file']) as f:
                content = f.read()
            try:
                json_end = content.index('\n}') + 2
                summary = json.loads(content[:json_end])
                
                print(f"{scenario['name']:<12} "
                      f"€{summary.get('net_benefit', 0):>13,.0f} "
                      f"€{summary.get('fcr_revenue', 0):>13,.0f} "
                      f"€{summary.get('peak_savings', 0):>13,.0f} "
                      f"€{summary.get('btm_savings', 0):>13,.0f}")
            except (json.JSONDecodeError, ValueError):
                print(f"{scenario['name']:<12} (summary parse error)")
    
    print("─" * 70)


def create_master_comparison_dashboard():
    """Create a master navigation dashboard with clickable links to each scenario."""
    import json
    
    print("\n" + "="*70)
    print("CREATING MASTER NAVIGATION DASHBOARD")
    print("="*70)
    
    # Load all scenario summaries
    scenarios = []
    for name in ['btm20', 'btm50', 'btm100']:
        pattern = f"dashboard_data_{name}_*.pkl.gz"
        files = list(RESULTS_DIR.glob(pattern))
        if files:
            data = load_scenario_data(files[0])
            
            # Get summary data
            port = data['financials']['portfolio']
            num_sites = len(data['site_configs'])
            total_capacity = sum(cfg.battery.E_max for cfg in data['site_configs'])
            
            scenarios.append({
                'name': name,
                'display_name': f"BTM {name.replace('btm', '')}%",
                'btm_ratio': float(name.replace('btm', '')) / 100,
                'ftm_ratio': 1 - float(name.replace('btm', '')) / 100,
                'net_benefit': port['net_benefit'],
                'fcr_revenue': port['fcr_revenue'],
                'peak_savings': port['peak_savings'],
                'btm_savings': port['btm_savings'],
                'degradation': port['degradation_cost'],
                'num_sites': num_sites,
                'total_capacity': total_capacity,
                'dashboard_file': f"dashboard_{name}.html",
                'site_dashboard': f"dashboard_{name}_Site_01.html",
            })
    
    if not scenarios:
        print("❌ No scenarios found!")
        return False
    
    # Sort by net benefit (best first)
    scenarios.sort(key=lambda x: x['net_benefit'], reverse=True)
    
    # Find best scenario
    best = scenarios[0]
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VPP Scenario Analysis - Master Dashboard</title>
    <style>
        :root {{
            --primary: #1a1a2e;
            --secondary: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --success: #00b894;
            --warning: #fdcb6e;
            --text: #edf2f7;
            --text-muted: #a0aec0;
            --card-bg: rgba(255,255,255,0.05);
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
            min-height: 100vh;
            color: var(--text);
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 50px;
        }}
        
        h1 {{
            font-size: 3em;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #fff, #00b894);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            font-size: 1.2em;
            color: var(--text-muted);
            margin-bottom: 20px;
        }}
        
        .summary-bar {{
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
            margin-bottom: 40px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }}
        
        .summary-item {{
            text-align: center;
        }}
        
        .summary-value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--success);
        }}
        
        .summary-value.negative {{
            color: var(--highlight);
        }}
        
        .summary-label {{
            font-size: 0.9em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .scenarios-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .scenario-card {{
            background: var(--card-bg);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .scenario-card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            border-color: var(--success);
        }}
        
        .scenario-card.best {{
            border: 2px solid var(--success);
        }}
        
        .scenario-card.best::before {{
            content: '🏆 BEST';
            position: absolute;
            top: 15px;
            right: 15px;
            background: var(--success);
            color: var(--primary);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 700;
        }}
        
        .scenario-card.negative::before {{
            content: '⚠️ LOSS';
            position: absolute;
            top: 15px;
            right: 15px;
            background: var(--highlight);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 700;
        }}
        
        .card-header {{
            margin-bottom: 25px;
        }}
        
        .card-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .card-subtitle {{
            color: var(--text-muted);
            font-size: 0.95em;
        }}
        
        .net-benefit {{
            font-size: 2.5em;
            font-weight: 700;
            margin: 20px 0;
            text-align: center;
        }}
        
        .net-benefit.positive {{
            color: var(--success);
        }}
        
        .net-benefit.negative {{
            color: var(--highlight);
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .metric {{
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 10px;
        }}
        
        .metric-label {{
            font-size: 0.8em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-value {{
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 5px;
        }}
        
        .metric-value.fcr {{ color: #9b59b6; }}
        .metric-value.btm {{ color: #27ae60; }}
        .metric-value.peak {{ color: #f39c12; }}
        .metric-value.deg {{ color: #e74c3c; }}
        
        .card-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .btn {{
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, var(--success), #00a381);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0,184,148,0.4);
        }}
        
        .btn-secondary {{
            background: rgba(255,255,255,0.1);
            color: var(--text);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .btn-secondary:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        .ratio-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
            display: flex;
        }}
        
        .ratio-btm {{
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            transition: width 0.5s ease;
        }}
        
        .ratio-ftm {{
            background: linear-gradient(90deg, #9b59b6, #8e44ad);
            transition: width 0.5s ease;
        }}
        
        .ratio-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: var(--text-muted);
        }}
        
        footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: var(--text-muted);
        }}
        
        .insights {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 30px;
            margin-top: 30px;
            backdrop-filter: blur(10px);
        }}
        
        .insights h2 {{
            margin-bottom: 20px;
            color: var(--success);
        }}
        
        .insight-item {{
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .insight-item:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔋 VPP Scenario Analysis</h1>
            <p class="subtitle">Multi-Battery Virtual Power Plant Optimization Results</p>
            <p class="subtitle">{scenarios[0]['num_sites']} Sites • {scenarios[0]['total_capacity']:,.0f} kWh Total Capacity • Full Year 2024</p>
        </header>
        
        <div class="summary-bar">
            <div class="summary-item">
                <div class="summary-value">{len(scenarios)}</div>
                <div class="summary-label">Scenarios</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">€{best['net_benefit']:,.0f}</div>
                <div class="summary-label">Best Net Benefit</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{best['display_name']}</div>
                <div class="summary-label">Optimal Strategy</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">€{best['fcr_revenue']:,.0f}</div>
                <div class="summary-label">Max FCR Revenue</div>
            </div>
        </div>
        
        <div class="scenarios-grid">
"""
    
    # Add scenario cards
    for i, s in enumerate(scenarios):
        is_best = (i == 0)
        is_negative = s['net_benefit'] < 0
        
        card_class = "scenario-card"
        if is_best:
            card_class += " best"
        elif is_negative:
            card_class += " negative"
        
        benefit_class = "positive" if s['net_benefit'] >= 0 else "negative"
        
        html += f"""
            <div class="{card_class}">
                <div class="card-header">
                    <h3 class="card-title">{s['display_name']}</h3>
                    <p class="card-subtitle">{s['btm_ratio']*100:.0f}% Behind-the-Meter • {s['ftm_ratio']*100:.0f}% FCR Market</p>
                </div>
                
                <div class="ratio-bar">
                    <div class="ratio-btm" style="width: {s['btm_ratio']*100}%"></div>
                    <div class="ratio-ftm" style="width: {s['ftm_ratio']*100}%"></div>
                </div>
                <div class="ratio-labels">
                    <span>🟢 BTM {s['btm_ratio']*100:.0f}%</span>
                    <span>🟣 FTM {s['ftm_ratio']*100:.0f}%</span>
                </div>
                
                <div class="net-benefit {benefit_class}">
                    €{s['net_benefit']:,.0f}
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">FCR Revenue</div>
                        <div class="metric-value fcr">€{s['fcr_revenue']:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">BTM Savings</div>
                        <div class="metric-value btm">€{s['btm_savings']:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Peak Savings</div>
                        <div class="metric-value peak">€{s['peak_savings']:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Degradation</div>
                        <div class="metric-value deg">-€{s['degradation']:,.0f}</div>
                    </div>
                </div>
                
                <div class="card-actions">
                    <a href="{s['dashboard_file']}" class="btn btn-primary">📊 Open Dashboard</a>
                    <a href="{s['site_dashboard']}" class="btn btn-secondary">📈 Site Details</a>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="insights">
            <h2>💡 Key Insights</h2>
            <div class="insight-item">
                <strong>FCR Dominates Value Creation:</strong> With 80% FTM allocation (BTM 20%), FCR revenue of €344,600 drives the highest net benefit. The minimum 1 MW bid requirement is easily met with aggregated battery capacity.
            </div>
            <div class="insight-item">
                <strong>Pure BTM Strategy Loses Money:</strong> With 100% BTM allocation, there's no FCR participation. The €15,789 in energy arbitrage savings cannot offset the €32,179 degradation cost, resulting in a net loss.
            </div>
            <div class="insight-item">
                <strong>Optimal Balance:</strong> A 20-50% BTM ratio provides the best trade-off between local load optimization and FCR market participation. The 50/50 split still generates €427,966 in net benefit.
            </div>
        </div>
        
        <footer>
            <p>Generated by VPP Optimization Dashboard • {scenarios[0]['num_sites']} Sites • Full Year 2024</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Click on any scenario card to view the detailed interactive dashboard
            </p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Save
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / "master_dashboard.html"
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Master navigation dashboard saved to: {output_file.name}")
    print("\n🌐 Open outputs/master_dashboard.html in your browser to navigate all scenarios!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="View and visualize saved optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_results.py                  # Generate all dashboards
  python view_results.py --list           # List available results
  python view_results.py --scenario btm50 # Generate for specific scenario
  python view_results.py --compare        # Show comparison table
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available scenario results'
    )
    
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default=None,
        help='Generate dashboard for specific scenario (e.g., btm20, btm50, btm100)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Show comparison table of all scenarios'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate dashboards for all scenarios'
    )
    
    parser.add_argument(
        '--master', '-m',
        action='store_true',
        help='Generate master comparison dashboard across all scenarios'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_scenarios()
    elif args.compare:
        create_comparison_overview()
    elif args.master:
        create_master_comparison_dashboard()
    elif args.scenario:
        generate_dashboard_for_scenario(args.scenario)
    elif args.all:
        generate_all_dashboards()
    else:
        # Default: list, compare, and generate master dashboard
        create_comparison_overview()
        create_master_comparison_dashboard()


if __name__ == "__main__":
    main()

