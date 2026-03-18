# Carbon Emission Tracker for ADAS Models (Using CodeCarbon)

This script uses the **CodeCarbon library** to measure actual carbon emissions from running ADAS (Advanced Driver Assistance Systems) models, then calculates the environmental savings from using these systems in freight trucks.

## What It Does

### 1. **Measures Actual Model Emissions (using CodeCarbon)**
- Real-time carbon tracking during model inference
- Energy consumption measurements
- GPU/CPU power monitoring
- Emissions per inference calculation

### 2. **Calculates ADAS Carbon Savings**
- Fuel efficiency improvements (8% average)
- Accident reduction benefits (21% fewer accidents)
- Idle time reduction (12%)
- Optimal routing improvements (5%)

### 3. **Net Environmental Impact**
- Total savings minus model emissions
- ROI ratio (environmental return on investment)
- Payback period analysis

## Models Analyzed

The script analyzes all 4 models in your checkpoints folder:
- **DDRNet-23s_cityscapes** (22.98 MB)
- **DDRNet-23s_sydneyscapes** (22.98 MB)
- **P2AT-M_cityscapes** (91.97 MB)
- **P2AT-M_sydneyscapes** (91.97 MB)

## How to Run

### Quick Start (Recommended)

Simply run the CodeCarbon-based analysis:

```bash
py run_codecarbon_analysis.py
```

This will:
- Run actual inference on your models with CodeCarbon tracking
- Measure real carbon emissions during model execution
- Calculate savings across different fleet sizes
- Generate CSV files with detailed data
- Create a human-readable summary report
- Save everything to the `carbon_results/` folder

### Installation

First, install CodeCarbon and dependencies:

```bash
pip install codecarbon torch opencv-python numpy
```

Or using the Python launcher on Windows:

```bash
py -m pip install codecarbon torch opencv-python numpy
```

### How CodeCarbon Works

CodeCarbon is a library that tracks carbon emissions from machine learning code by:

1. **Monitoring Hardware**: Tracks CPU and GPU power consumption in real-time
2. **Measuring Duration**: Records how long your code runs
3. **Calculating Energy**: Computes energy used (kWh) based on hardware specs
4. **Estimating Emissions**: Converts energy to CO2 emissions using grid carbon intensity

**Key Benefits:**
- Real measurements, not just estimates
- Automatic GPU detection and monitoring
- Saves detailed emission logs to CSV
- Works on CPU and GPU systems

### Advanced Usage

For custom scenarios, you can modify and run the main calculator:

```bash
python carbon_calculator.py
```

Or import and use in your own scripts:

```python
from carbon_calculator import CarbonCalculator

# Create calculator
calculator = CarbonCalculator(output_dir="my_results")

# Run custom analysis
results = calculator.run_full_analysis(
    num_trucks_list=[100, 500, 1000, 5000, 10000],
    years_list=[1, 5, 10]
)

# Save results
calculator.save_results_to_csv(results)
calculator.generate_summary_report(results)
```

### Specific Calculations

You can also run specific calculations:

```python
from carbon_calculator import CarbonCalculator

calculator = CarbonCalculator()

# Calculate emissions for one model
emissions = calculator.calculate_model_inference_emissions(
    model_name="DDRNet-23s_cityscapes",
    fps=30,
    hours_per_day=8,
    days_per_year=250
)
print(f"Annual emissions: {emissions['annual_emissions_tons_co2']:.6f} tons CO2")

# Calculate ADAS savings for a fleet
savings = calculator.calculate_adas_savings(
    num_trucks=1000,
    years_of_operation=5
)
print(f"Total savings: {savings['total_savings_over_period_tons']:.2f} tons CO2")

# Calculate net impact
net_impact = calculator.calculate_net_impact(
    model_name="P2AT-M_cityscapes",
    num_trucks=1000,
    years_of_operation=5
)
print(f"Net impact: {net_impact['net_impact_tons']:.2f} tons CO2")
print(f"ROI: {net_impact['roi_ratio']:.1f}x")
```

## Output Files

After running, you'll find these files in the `carbon_results/` directory:

### 1. `codecarbon_analysis_YYYYMMDD_HHMMSS.csv`
Detailed CSV with all measurements including:
- Model specifications and sizes
- **Measured emissions per inference** (from CodeCarbon)
- Inference speed (inferences per second)
- Annual emission projections
- Fleet savings calculations
- Net impact analysis

### 2. `codecarbon_report_YYYYMMDD_HHMMSS.txt`
Human-readable summary report including:
- Models analyzed and their specifications
- **Measured emissions** from actual inference runs
- Net impact summary
- ADAS benefits breakdown

### 3. `emissions.csv` (CodeCarbon Raw Output)
Raw data from CodeCarbon including:
- Timestamp for each measurement
- CPU and GPU power consumption
- Memory usage
- Carbon emissions for each run
- Country and region information (for grid carbon intensity)

## Understanding the Results

### Key Metrics from CodeCarbon

| Metric | Description |
|--------|-------------|
| **emissions_per_inference_g** | Actual CO2 emitted per inference run (measured by CodeCarbon) |
| **inferences_per_second** | Processing speed measured during tracking |
| **total_emissions_kg** | Total emissions for the tracked inference run |
| **annual_emissions_tons_co2** | Projected annual emissions based on measurements |
| **total_annual_savings_tons** | Total CO2 saved by ADAS across the fleet per year |
| **net_impact_tons** | Net environmental impact (savings - measured emissions) |
| **roi_ratio** | Environmental ROI: how much CO2 saved per unit emitted |
| **is_net_positive** | True if the system provides net carbon savings |

### CodeCarbon Measurements

The script runs **100 inferences per model** with CodeCarbon tracking to measure:
- Actual energy consumption during inference
- CPU/GPU power draw
- Real-time carbon emissions
- Processing throughput

These measurements are then extrapolated to annual projections based on:
- 30 frames per second processing
- 8 hours per day operation
- 250 days per year

### Positive Net Impact

All models show **positive net environmental impact** because:
- ADAS fuel savings (~8-25% improvement) far exceed measured model inference emissions
- Accident reduction provides additional carbon savings
- Models are efficient (small size, optimized inference)

## Assumptions and References

The calculations are based on:
- **EPA**: 22.2 lbs CO2 per gallon of diesel fuel
- **Industry data**: 6.5 mpg average for heavy trucks
- **Research**: 5-15% fuel efficiency improvement from ADAS
- **US Grid**: 0.475 kg CO2 per kWh (national average)

### ADAS Benefits (Research-Based)
- Fuel efficiency: +8% (various studies)
- Accident reduction: -21% (IIHS research)
- Idle time reduction: -12% (fleet studies)
- Optimal routing: +5% (telematics data)

## Requirements

The script uses only standard Python libraries:
- `csv`, `os`, `pathlib`, `datetime` - Standard library
- `torch`, `psutil` - For optional hardware monitoring

Install if needed:
```bash
pip install torch psutil
```

## Troubleshooting

**Issue**: Script doesn't run
- Ensure you're in the project root directory
- Check Python version (3.7+ recommended)

**Issue**: No output files created
- Check permissions in the current directory
- Verify the `carbon_results/` folder can be created

**Issue**: Results seem incorrect
- Review assumptions in the code constants
- Adjust fleet size and time periods to match your use case

## Customization

You can easily customize the analysis by modifying constants in `CarbonCalculator`:

```python
# In carbon_calculator.py, modify these values:

# Fleet parameters
AVG_FREIGHT_FUEL_CONSUMPTION = 6.5  # mpg
AVG_ANNUAL_MILEAGE = 100000  # miles per year

# Grid carbon intensity (adjust for your region)
GRID_CARBON_INTENSITY = 0.475  # kg CO2/kWh (US average)
# For specific regions:
# Australia: 0.85, EU: 0.3, China: 0.6

# ADAS benefits (adjust based on your research)
ADAS_BENEFITS = {
    "fuel_efficiency_improvement": 0.08,
    "accident_reduction": 0.21,
    "idle_time_reduction": 0.12,
    "optimal_routing": 0.05,
}
```

## Support

For questions or issues:
1. Check the generated summary report for detailed explanations
2. Review the assumptions section in the code
3. Adjust parameters to match your specific scenario

---

**Generated by**: Carbon Calculator v1.0
**Last Updated**: 2026-03-18
