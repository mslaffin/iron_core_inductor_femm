# Iron Core Inductor Simulation with PyFEMM

A comprehensive workflow for simulating iron core inductors using pyfemm 0.1.3 (Finite Element Method Magnetics) with automation tools.

## Features

- **Configurable DXF Generation**: Automated creation of inductor geometry with iron core and wire cross-sections
- **Intelligent Wire Positioning**: Automatic calculation of wire positions based on winding volume and turn count
- **Complete FEMM Integration**: Scripted import, material properties, circuit setup, and simulation
- **Comprehensive Analysis**: Post-simulation analysis including inductance, losses, and field distribution
- **Flexible Configuration**: JSON-based configuration system for different inductor designs

## Components

### 1. DXF Generator (`dxf_generator.py`)
- Creates configurable DXF files with iron core geometry
- Calculates optimal wire cross-section positions
- Supports multiple wire layers and configurable winding volumes
- Generates construction lines and organized layers

### 2. FEMM Simulation (`femm_simulation.py`)
- Handles DXF import and geometry creation
- Defines material properties (iron core, copper wire, air)
- Configures circuit properties and boundary conditions
- Manages complete simulation workflow

### 3. Analysis Tools (`analysis_tools.py`)
- Calculates inductance from flux linkage
- Analyzes magnetic field distribution
- Computes core and copper losses
- Generates comprehensive plots and reports

### 4. Main Workflow (`inductor_workflow.py`)
- Integrates all components into a single workflow
- Command-line interface with configuration options
- Supports batch processing and parameter studies

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure FEMM is installed and accessible to pyfemm

## Usage

### Quick Start

Run with default configuration:
```bash
python inductor_workflow.py
```

### Custom Configuration

1. Create sample configuration files:
```bash
python inductor_workflow.py --create-samples
```

2. Run with custom configuration:
```bash
python inductor_workflow.py --config small_inductor_config.json
```

### Command Line Options

- `--config, -c`: Specify configuration file (JSON format)
- `--output-dxf, -o`: Output DXF filename (default: inductor.dxf)
- `--no-dxf`: Skip DXF generation (use existing geometry)
- `--save-config`: Save current configuration to file
- `--create-samples`: Create sample configuration files
- `--analysis-only`: Run analysis only (requires active FEMM solution)

### Configuration Format

Configuration files use JSON format with three main sections:

```json
{
  "geometry": {
    "core_outer_radius": 20.0,
    "core_inner_radius": 8.0,
    "wire_diameter": 0.8,
    "num_turns": 40,
    "winding_inner_radius": 8.5,
    "winding_outer_radius": 19.5,
    "air_gap_radius": 50.0
  },
  "materials": {
    "iron_mu_r": 4000.0,
    "copper_conductivity": 58e6
  },
  "circuit": {
    "current_amplitude": 1.0,
    "frequency": 0.0,
    "circuit_name": "MainCoil"
  }
}
```

## Example Workflows

### 1. Basic Inductor Design
```bash
# Generate geometry and run simulation
python inductor_workflow.py --output-dxf my_inductor.dxf

# Results: inductance calculation, field plots, analysis report
```

### 2. Parameter Study
```bash
# Create configurations for different designs
python inductor_workflow.py --create-samples

# Run simulations with different parameters
python inductor_workflow.py --config small_inductor_config.json
python inductor_workflow.py --config large_inductor_config.json
```

### 3. Custom Design
```python
from inductor_workflow import InductorWorkflow
from dxf_generator import InductorConfig

# Create custom configuration
config = InductorConfig(
    core_outer_radius=15.0,
    core_inner_radius=6.0,
    wire_diameter=0.6,
    num_turns=30
)

# Run workflow
workflow = InductorWorkflow()
workflow.config = config
workflow.run_complete_workflow()
```

## Output Files

The simulation generates several output files:

- `inductor.dxf`: Geometry file for FEMM import
- `analysis_report.txt`: Comprehensive analysis results
- `inductor_field_analysis.png`: Magnetic field distribution plots
- Configuration files (when using `--save-config`)

## Analysis Results

The analysis provides:

### Electrical Properties
- **Inductance**: Calculated from flux linkage and current
- **Flux Linkage**: Total flux linking the coil
- **Stored Energy**: Magnetic energy stored in the field

### Magnetic Field Analysis
- **Maximum Flux Density**: Peak magnetic field strength
- **Core Flux Density**: Average field in the iron core
- **Field Distribution**: 2D maps and vector plots

### Loss Analysis
- **Copper Losses**: I²R losses in the windings
- **Core Losses**: Hysteresis and eddy current losses
- **Efficiency Metrics**: Performance indicators

### Visualization
- Flux density magnitude contours
- Vector field plots
- Radial field distribution
- Energy density maps

## Customization

### Adding New Materials
Modify the `MaterialProperties` class in `femm_simulation.py`:

```python
@dataclass
class MaterialProperties:
    # Add custom material properties
    custom_material_mu_r: float = 1000.0
    custom_material_conductivity: float = 1e6
```

### Custom Analysis Functions
Extend the `MagneticAnalyzer` class in `analysis_tools.py`:

```python
def calculate_custom_metric(self):
    # Add custom analysis routines
    pass
```

### Geometry Modifications
Extend the `DXFGenerator` class in `dxf_generator.py` for custom shapes:

```python
def add_custom_geometry(self):
    # Add custom geometric features
    pass
```

## Troubleshooting

### Common Issues

1. **FEMM Communication Error**: Ensure FEMM is properly installed and pyfemm can access it
2. **DXF Import Failure**: Check DXF file format and layer names
3. **Convergence Problems**: Adjust mesh density or boundary conditions
4. **Memory Issues**: Reduce simulation domain size or mesh refinement

### Debugging

Enable verbose output by modifying the simulation scripts or add debug prints as needed.

## Requirements

- Python 3.7+
- pyfemm 0.1.3
- ezdxf >= 1.0.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- FEMM (Finite Element Method Magnetics) software

## License

This project is provided as-is for educational and research purposes.