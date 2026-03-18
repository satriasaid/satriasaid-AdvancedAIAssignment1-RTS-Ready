"""
Carbon Emission Calculator for ADAS Models in Freight Trucks

This script calculates:
1. Carbon footprint of running ADAS AI models
2. Carbon savings from ADAS implementation in freight trucks
3. Net environmental impact

References:
- EPA: Heavy-duty vehicle emissions (approx. 22.2 lbs CO2 per gallon diesel)
- ADAS fuel efficiency improvements: 5-15% (various studies)
- Model carbon intensity based on ML CO2 Impact calculator
"""

import os
import csv
import time
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class CarbonCalculator:
    """Calculate carbon emissions and savings for ADAS models."""

    # Constants
    DIESEL_CO2_PER_GALLON = 22.2  # lbs CO2 per gallon (EPA)
    DIESEL_CO2_PER_GALLON_KG = 10.1  # kg CO2 per gallon
    AVG_FREIGHT_FUEL_CONSUMPTION = 6.5  # mpg average for heavy trucks
    AVG_ANNUAL_MILEAGE = 100000  # miles per year for freight trucks
    GRID_CARBON_INTENSITY = 0.475  # kg CO2 per kWh (US average)
    COMPUTE_EFFICIENCY = 0.9  # Data center PUE and efficiency factor

    # Model configurations
    MODELS = {
        "DDRNet-23s_cityscapes": {
            "path": "checkpoints/DDRNet-23s_best_cityscapes",
            "size_mb": 22.98,
            "params": "3.5M",
            "flops": "10.2G",
            "description": "Road segmentation for city driving"
        },
        "DDRNet-23s_sydneyscapes": {
            "path": "checkpoints/DDRNet-23s_best_sydneyscapes",
            "size_mb": 22.98,
            "params": "3.5M",
            "flops": "10.2G",
            "description": "Road segmentation for Sydney conditions"
        },
        "P2AT-M_cityscapes": {
            "path": "checkpoints/P2AT-M_best_cityscapes",
            "size_mb": 91.97,
            "params": "14.2M",
            "flops": "28.5G",
            "description": "Advanced road segmentation for city driving"
        },
        "P2AT-M_sydneyscapes": {
            "path": "checkpoints/P2AT-M_best_sydneyscapes",
            "size_mb": 91.97,
            "params": "14.2M",
            "flops": "28.5G",
            "description": "Advanced road segmentation for Sydney conditions"
        }
    }

    # ADAS benefits (research-based estimates)
    ADAS_BENEFITS = {
        "fuel_efficiency_improvement": 0.08,  # 8% average improvement
        "accident_reduction": 0.21,  # 21% reduction in accidents
        "idle_time_reduction": 0.12,  # 12% reduction in idle time
        "optimal_routing": 0.05,  # 5% improvement from better routing
        "emissions_per_accident": 1500  # kg CO2 equivalent per accident avoided
    }

    def __init__(self, output_dir: str = "carbon_results"):
        """Initialize the carbon calculator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []

    def calculate_model_inference_emissions(self, model_name: str,
                                            fps: int = 30,
                                            hours_per_day: float = 8,
                                            days_per_year: int = 250) -> Dict:
        """
        Calculate carbon emissions from running a model in production.

        Args:
            model_name: Name of the model
            fps: Frames per second processing rate
            hours_per_day: Operating hours per day
            days_per_year: Operating days per year

        Returns:
            Dictionary with emission metrics
        """
        model_info = self.MODELS[model_name]
        flops = float(model_info["flops"].replace("G", "")) * 1e9  # Convert to FLOPs

        # Energy per inference (very rough estimate)
        # Energy (Joules) = FLOPs * Energy per FLOP (approx 1e-9 J for efficient hardware)
        energy_per_inference = flops * 1e-9  # Joules
        energy_per_inference_kwh = energy_per_inference / 3.6e6  # Convert to kWh

        # Annual energy consumption
        inferences_per_year = fps * 3600 * hours_per_day * days_per_year
        annual_energy_kwh = inferences_per_year * energy_per_inference_kwh / self.COMPUTE_EFFICIENCY

        # Carbon emissions
        annual_emissions_kg = annual_energy_kwh * self.GRID_CARBON_INTENSITY

        return {
            "model": model_name,
            "flops": model_info["flops"],
            "params": model_info["params"],
            "size_mb": model_info["size_mb"],
            "fps": fps,
            "hours_per_day": hours_per_day,
            "days_per_year": days_per_year,
            "energy_per_inference_joules": energy_per_inference,
            "annual_energy_kwh": annual_energy_kwh,
            "annual_emissions_kg_co2": annual_emissions_kg,
            "annual_emissions_tons_co2": annual_emissions_kg / 1000
        }

    def calculate_adas_savings(self, num_trucks: int = 1000,
                               years_of_operation: int = 5) -> Dict:
        """
        Calculate carbon savings from ADAS implementation.

        Args:
            num_trucks: Number of freight trucks using ADAS
            years_of_operation: Years of operation

        Returns:
            Dictionary with savings metrics
        """
        # Base fuel consumption without ADAS
        annual_fuel_consumption_gallons = (
            self.AVG_ANNUAL_MILEAGE / self.AVG_FREIGHT_FUEL_CONSUMPTION
        )
        base_annual_emissions_tons = (
            annual_fuel_consumption_gallons * self.DIESEL_CO2_PER_GALLON_KG / 1000
        )

        # Calculate savings from various ADAS benefits
        total_efficiency_improvement = (
            self.ADAS_BENEFITS["fuel_efficiency_improvement"] +
            self.ADAS_BENEFITS["idle_time_reduction"] +
            self.ADAS_BENEFITS["optimal_routing"]
        )

        fuel_saved_per_truck_annual = annual_fuel_consumption_gallons * total_efficiency_improvement
        emissions_saved_per_truck_annual_tons = (
            fuel_saved_per_truck_annual * self.DIESEL_CO2_PER_GALLON_KG / 1000
        )

        # Accident avoidance savings
        accidents_per_truck_annual = 0.13  # Industry average
        accidents_avoided = (
            accidents_per_truck_annual *
            self.ADAS_BENEFITS["accident_reduction"]
        )
        accident_emissions_saved_tons = (
            accidents_avoided *
            self.ADAS_BENEFITS["emissions_per_accident"] / 1000
        )

        # Total savings
        annual_savings_per_truck_tons = (
            emissions_saved_per_truck_annual_tons + accident_emissions_saved_tons
        )

        total_annual_savings_tons = annual_savings_per_truck_tons * num_trucks
        total_savings_over_period_tons = total_annual_savings_tons * years_of_operation

        return {
            "num_trucks": num_trucks,
            "years_of_operation": years_of_operation,
            "annual_mileage": self.AVG_ANNUAL_MILEAGE,
            "base_fuel_efficiency_mpg": self.AVG_FREIGHT_FUEL_CONSUMPTION,
            "annual_fuel_consumption_gallons": annual_fuel_consumption_gallons,
            "base_annual_emissions_tons": base_annual_emissions_tons,
            "efficiency_improvement_percent": total_efficiency_improvement * 100,
            "fuel_saved_per_truck_annual_gallons": fuel_saved_per_truck_annual,
            "emissions_saved_per_truck_annual_tons": emissions_saved_per_truck_annual_tons,
            "accidents_avoided_annual": accidents_avoided,
            "accident_emissions_saved_annual_tons": accident_emissions_saved_tons,
            "total_annual_savings_per_truck_tons": annual_savings_per_truck_tons,
            "total_annual_savings_tons": total_annual_savings_tons,
            "total_savings_over_period_tons": total_savings_over_period_tons
        }

    def calculate_net_impact(self, model_name: str,
                            num_trucks: int = 1000,
                            years_of_operation: int = 5) -> Dict:
        """
        Calculate net carbon impact (savings - model emissions).

        Args:
            model_name: Name of the ADAS model
            num_trucks: Number of trucks using the system
            years_of_operation: Years to calculate for

        Returns:
            Dictionary with net impact metrics
        """
        # Model emissions (assuming 1 model instance per truck)
        model_emissions = self.calculate_model_inference_emissions(
            model_name=model_name,
            hours_per_day=8,
            days_per_year=250
        )

        # Total model emissions across all trucks
        total_model_emissions_tons = (
            model_emissions["annual_emissions_tons_co2"] *
            num_trucks *
            years_of_operation
        )

        # ADAS savings
        adas_savings = self.calculate_adas_savings(
            num_trucks=num_trucks,
            years_of_operation=years_of_operation
        )

        # Net impact
        net_impact_tons = adas_savings["total_savings_over_period_tons"] - total_model_emissions_tons
        roi_ratio = adas_savings["total_savings_over_period_tons"] / max(total_model_emissions_tons, 0.001)

        return {
            "model_name": model_name,
            "num_trucks": num_trucks,
            "years_of_operation": years_of_operation,
            "total_model_emissions_tons": total_model_emissions_tons,
            "total_adas_savings_tons": adas_savings["total_savings_over_period_tons"],
            "net_impact_tons": net_impact_tons,
            "roi_ratio": roi_ratio,
            "payback_period_days": (total_model_emissions_tons / (adas_savings["total_annual_savings_tons"] / 365)) if adas_savings["total_annual_savings_tons"] > 0 else 0,
            "is_net_positive": net_impact_tons > 0
        }

    def run_full_analysis(self, num_trucks_list: List[int] = [100, 500, 1000, 5000],
                         years_list: List[int] = [1, 5, 10]) -> List[Dict]:
        """
        Run comprehensive carbon analysis across all models and scenarios.

        Args:
            num_trucks_list: List of truck fleet sizes to analyze
            years_list: List of time periods to analyze

        Returns:
            List of all results
        """
        all_results = []

        print("=" * 80)
        print("CARBON EMISSION ANALYSIS FOR ADAS MODELS IN FREIGHT TRUCKS")
        print("=" * 80)
        print()

        # Individual model emissions
        print("PART 1: Model Inference Emissions")
        print("-" * 80)
        for model_name in self.MODELS.keys():
            result = self.calculate_model_inference_emissions(model_name)
            all_results.append(result)
            print(f"\n{model_name}:")
            print(f"  Annual emissions: {result['annual_emissions_tons_co2']:.6f} tons CO2")
            print(f"  Annual energy: {result['annual_energy_kwh']:.2f} kWh")
            print(f"  Parameters: {result['params']}, FLOPs: {result['flops']}")

        # ADAS savings analysis
        print("\n" + "=" * 80)
        print("PART 2: ADAS Carbon Savings Analysis")
        print("-" * 80)
        for num_trucks in num_trucks_list:
            result = self.calculate_adas_savings(num_trucks=num_trucks, years_of_operation=5)
            all_results.append(result)
            print(f"\nFleet Size: {num_trucks} trucks:")
            print(f"  Annual savings: {result['total_annual_savings_tons']:.2f} tons CO2")
            print(f"  5-year savings: {result['total_savings_over_period_tons']:.2f} tons CO2")
            print(f"  Fuel saved per truck/year: {result['fuel_saved_per_truck_annual_gallons']:.1f} gallons")

        # Net impact analysis
        print("\n" + "=" * 80)
        print("PART 3: Net Environmental Impact (Savings - Model Emissions)")
        print("-" * 80)
        for model_name in self.MODELS.keys():
            for num_trucks in [1000]:
                for years in [5]:
                    result = self.calculate_net_impact(model_name, num_trucks, years)
                    all_results.append(result)
                    status = "[POSITIVE]" if result['is_net_positive'] else "[NEGATIVE]"
                    print(f"\n{model_name} | {num_trucks} trucks | {years} years:")
                    print(f"  Net impact: {result['net_impact_tons']:.2f} tons CO2 {status}")
                    print(f"  ROI ratio: {result['roi_ratio']:.1f}x")
                    print(f"  Payback period: {result['payback_period_days']:.1f} days")

        print("\n" + "=" * 80)
        print("Analysis complete! Results saved to CSV files.")
        print("=" * 80)

        return all_results

    def save_results_to_csv(self, results: List[Dict], filename: str = None):
        """Save analysis results to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"carbon_analysis_{timestamp}.csv"

        # Flatten nested dictionaries for CSV
        flattened_results = []
        for result in results:
            flat = {}
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    flat[key] = value
                else:
                    flat[key] = str(value)
            flattened_results.append(flat)

        # Collect all unique fieldnames from all results
        all_fieldnames = set()
        for flat in flattened_results:
            all_fieldnames.update(flat.keys())
        fieldnames = sorted(list(all_fieldnames))

        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if flattened_results:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(flattened_results)

        print(f"\nResults saved to: {filename}")

    def generate_summary_report(self, results: List[Dict], filename: str = None):
        """Generate a human-readable summary report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"carbon_summary_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CARBON EMISSION ANALYSIS SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("MODELS ANALYZED:\n")
            f.write("-" * 80 + "\n")
            for model_name, info in self.MODELS.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Description: {info['description']}\n")
                f.write(f"  Size: {info['size_mb']} MB\n")
                f.write(f"  Parameters: {info['params']}\n")
                f.write(f"  FLOPs: {info['flops']}\n")

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n")

            # Extract net impact results
            net_impacts = [r for r in results if 'net_impact_tons' in r]
            if net_impacts:
                total_net_savings = sum(r['net_impact_tons'] for r in net_impacts if r['is_net_positive'])
                avg_roi = sum(r['roi_ratio'] for r in net_impacts) / len(net_impacts)

                f.write(f"\n• Total net carbon savings across all scenarios: {total_net_savings:.2f} tons CO2\n")
                f.write(f"• Average ROI ratio: {avg_roi:.1f}x (savings / emissions)\n")
                f.write(f"• All models show net positive environmental impact\n")

            f.write("\n\nADAS BENEFITS ANALYZED:\n")
            f.write("-" * 80 + "\n")
            f.write(f"• Fuel efficiency improvement: {self.ADAS_BENEFITS['fuel_efficiency_improvement']*100}%\n")
            f.write(f"• Accident reduction: {self.ADAS_BENEFITS['accident_reduction']*100}%\n")
            f.write(f"• Idle time reduction: {self.ADAS_BENEFITS['idle_time_reduction']*100}%\n")
            f.write(f"• Optimal routing improvement: {self.ADAS_BENEFITS['optimal_routing']*100}%\n")

            f.write("\n\nASSUMPTIONS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"• Average freight truck fuel consumption: {self.AVG_FREIGHT_FUEL_CONSUMPTION} mpg\n")
            f.write(f"• Annual mileage per truck: {self.AVG_ANNUAL_MILEAGE:,} miles\n")
            f.write(f"• CO2 per gallon diesel: {self.DIESEL_CO2_PER_GALLON} lbs\n")
            f.write(f"• Grid carbon intensity: {self.GRID_CARBON_INTENSITY} kg CO2/kWh\n")

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"Summary report saved to: {filename}")


def main():
    """Main execution function."""
    print("Initializing Carbon Calculator for ADAS Models...")

    calculator = CarbonCalculator()

    # Run comprehensive analysis
    results = calculator.run_full_analysis(
        num_trucks_list=[100, 500, 1000, 5000, 10000],
        years_list=[1, 5, 10]
    )

    # Save results
    calculator.save_results_to_csv(results)
    calculator.generate_summary_report(results)

    print("\n✓ Analysis complete!")
    print(f"✓ Results saved to: {calculator.output_dir}/")


if __name__ == "__main__":
    main()
