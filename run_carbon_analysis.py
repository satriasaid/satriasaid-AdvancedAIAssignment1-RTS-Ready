"""
Quick start script for carbon analysis.
Run this to generate the carbon emission reports.
"""

from carbon_calculator import CarbonCalculator


def main():
    """Run quick carbon analysis."""
    print("=" * 80)
    print("ADAS CARBON EMISSION ANALYZER")
    print("=" * 80)
    print("\nThis will analyze carbon footprint and savings for your ADAS models.")
    print("Please wait...\n")

    # Create calculator
    calculator = CarbonCalculator(output_dir="carbon_results")

    # Quick analysis with standard scenarios
    print("Running analysis for standard scenarios...")
    results = calculator.run_full_analysis(
        num_trucks_list=[100, 1000, 5000],
        years_list=[5]
    )

    # Save all results
    calculator.save_results_to_csv(results)

    # Generate summary report
    calculator.generate_summary_report(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • carbon_results/carbon_analysis_*.csv - Detailed data")
    print("  • carbon_results/carbon_summary_*.txt - Summary report")
    print("\nOpen these files to see the full analysis results.")


if __name__ == "__main__":
    main()
