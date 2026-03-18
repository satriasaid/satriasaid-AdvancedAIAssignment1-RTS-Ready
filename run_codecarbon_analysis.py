"""
Quick start script for CodeCarbon-based carbon analysis.
Run this to measure actual carbon emissions from your ADAS models.
"""

from carbon_tracker_with_codecarbon import ADASCarbonTracker


def main():
    """Run CodeCarbon-based carbon analysis."""
    print("="*80)
    print("ADAS CARBON TRACKER - USING CODECARBON")
    print("="*80)
    print("\nThis will measure actual carbon emissions from your models")
    print("and calculate savings from ADAS implementation in freight trucks.\n")

    # Create tracker
    tracker = ADASCarbonTracker(output_dir="carbon_results")

    # Run analysis
    results = tracker.run_full_analysis(
        models=None,  # All models in checkpoints/
        num_trucks_list=[100, 1000, 5000],
        years=5
    )

    # Save results
    tracker.save_to_csv(results)
    tracker.generate_report(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - carbon_results/codecarbon_analysis_*.csv")
    print("  - carbon_results/codecarbon_report_*.txt")
    print("  - carbon_results/emissions.csv (CodeCarbon raw data)")
    print("\nOpen these files to see your carbon footprint and savings analysis.")


if __name__ == "__main__":
    main()
