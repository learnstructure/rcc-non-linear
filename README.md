# RCC Non-Linear Analysis Toolkit

A comprehensive Python package for nonlinear analysis and modeling of reinforced concrete (RCC) structures, particularly focused on RC columns. This toolkit provides tools for concrete material modeling using Mander's confined concrete model, OpenSees integration, and pushover analysis capabilities.

## Features

- **Concrete Material Modeling**: Implements Mander's confined concrete model for both rectangular and circular cross-sections
- **Steel Material Modeling**: Bilinear and advanced steel constitutive models with strain hardening
- **OpenSees Integration**: Automated OpenSees model generation for complex structural analyses
- **Section Analysis**: Moment-curvature (M-φ) analysis for rectangular and circular sections
- **Pushover Analysis**: Nonlinear static pushover analysis capabilities
- **Reporting**: Automated markdown and PDF report generation for analysis results
- **Cross-section Support**: Full support for both rectangular and circular reinforced concrete columns

## Installation

### Option 1: Using conda (Recommended)

Conda is cross-platform and handles all dependencies automatically. First, install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download).

Then create and activate the environment:

```bash
conda env create -f environment.yml
conda activate rcc_env
```

Install the package:
```bash
pip install git+https://github.com/learnstructure/rcc-non-linear.git
```

### Option 2: Using pip (Without Conda)

First, ensure you have Python 3.8 installed. If not, install it from [python.org](https://www.python.org/downloads/release/python-3810/) or using a package manager:

Create and activate a virtual environment:
```bash
python3.8 -m venv rcc_env
source rcc_env/bin/activate   # On Windows: rcc_env\Scripts\activate
```

Install the package with OpenSees support:

```bash
pip install "git+https://github.com/learnstructure/rcc-non-linear.git#egg=rcc-non-linear[opensees]"
```

**Note**: If you're using Python > 3.8 and don't need OpenSees integration, you can install without OpenSees:

```bash
python -m venv rcc_env
source rcc_env/bin/activate   # On Windows: rcc_env\Scripts\activate
```
Install the package:
```bash
pip install git+https://github.com/learnstructure/rcc-non-linear.git
```

### Option 3: Running on Google Colab
```bash
pip install openseespy
pip install opsvis
pip install git+https://github.com/learnstructure/rcc-non-linear.git
```

### Dependencies

- Python ≥ 3.8
- numpy
- scipy
- matplotlib
- pandas
- openseespy (optional, for OpenSees integration)
- opsvis (optional, for OpenSees visualization)
- reportlab

## Project Structure
```
rcc_non_linear/
├── concrete_models/          # Concrete material models
│   ├── mander_model.py       # Mander's confined concrete model
│   └── data/
│       └── rect_conf_k.csv   # Confinement effectiveness lookup table
├── opensees_model/           # OpenSees model generation
│   ├── model.py              # Main model class
│   ├── rect_section.py       # Rectangular section
│   ├── circ_section.py       # Circular section
│   ├── material.py           # Material definitions
│   ├── m_phi.py              # Moment-curvature analysis
│   ├── pushover.py           # Pushover analysis
│   └── gravity.py            # Gravity load handling
├── utils/                    # Utility functions
│   ├── helper.py             # Helper functions and interpolation
│   └── report.py             # Report generation
├── config.py                 # Configuration settings
└── __init__.py               # Package initialization

examples/                      # Example scripts
├── rect_ke.py                # Rectangular column ke calculation
├── rect_M_phi.py             # Rectangular column M-φ analysis
├── circ_M_phi.py             # Circular column M-φ analysis
├── rect_pushover.py          # Rectangular column pushover
├── circ_pushover.py          # Circular column pushover
└── ...

full_versions/                # Complete standalone implementations
```

## Quick Start

### Example 1: Rectangular Column Concrete Model

```python
from rcc_non_linear import RectConcreteMander

# Define rectangular column concrete properties
rect_concrete = RectConcreteMander(
    fc_prime=5.075,    # ksi - unconfined concrete strength
    B=27.558,          # in - width
    H=13.779,          # in - depth
    cover=1.77,        # in - concrete cover
    dh=0.248,          # in - transverse bar diameter
    sh=11.811,         # in - transverse bar spacing
    fyh=68,            # ksi - transverse bar yield strength
    esm=0.1,           # ultimate strain
    nx=2, ny=2,        # number of transverse bar legs
    ke=0.75            # confinement effectiveness coefficient
)

print(f"Confined strength: {rect_concrete.fcc_prime} ksi")
print(f"Strain at peak: {rect_concrete.ecc}")
```

### Example 2: Rectangular Column Model with OpenSees

```python
from rcc_non_linear import Model

# Define column properties
props = {
    'fc': 5.0,                  # ksi - concrete strength
    'L': 144,                   # in - column length
    'cover': 1.75,              # in - concrete cover
    'B': 27.5, 'H': 13.75,      # in - section dimensions
    'nBarsTop': 6, 'dbTop': 1.0,
    'nBarsBot': 6, 'dbBot': 1.0,
    'fy': 70, 'fu': 120,        # ksi - steel properties
    'fyh': 68,                  # ksi - transverse steel
    'dh': 0.25,                 # in - transverse bar diameter
    'sh': 12.0,                 # in - transverse spacing
    'P': 500,                   # kip - axial load
}

# Create model
model = Model(props)
model.create_model()

# Run pushover analysis
model.pushover(n_steps=100, d_roof=50, direction='x')

# Generate report
model.create_report()
```

### Example 3: Circular Column Model with OpenSees

```python
from rcc_non_linear import Model
from rcc_non_linear.utils.helper import plot_response, plot_response_multi
col_props = {
    'fc': 6.1,
    'D': 48, 'L': 324, 'cover': 2,
    'nBars': 18, 'db': 1.41,
    'fy': 75.2, 'fu': 102.4, 'Es': 29000, 'Esh': 1247, 'e_sh': 0.005, 'e_ult': 0.122,
    'dh':0.888, 'sh':6, 'fyh':54.8, 'esm':0.125,
    'P_axial': 570, 'rupture_limit': 0.15
}
model = Model(col_props)

results_df, bilinear_df, yield_step = model.run_pushover_analysis()
```

## Key Classes

### RectConcreteMander
Implements Mander's confined concrete model for rectangular sections.

**Parameters:**
- `fc_prime`: Unconfined concrete compressive strength
- `B`, `H`: Section dimensions
- `cover`: Concrete cover
- `dh`: Transverse bar diameter
- `sh`: Transverse bar spacing
- `fyh`: Transverse bar yield strength
- `esm`: Ultimate strain
- `nx`, `ny`: Number of transverse bar legs
- `ke`: Confinement effectiveness coefficient

**Attributes:**
- `fcc_prime`: Confined concrete peak stress
- `ecc`: Strain at peak stress
- `k`: Enhancement factor
- `ecu`: Ultimate compressive strain

### CircConcreteMander
Similar to `RectConcreteMander` but for circular sections.

### Model
Main class for creating complete RC column models.

**Methods:**
- `create_model()`: Initialize OpenSees model
- `m_phi_analysis()`: Perform moment-curvature analysis
- `pushover()`: Run nonlinear pushover analysis
- `create_report()`: Generate markdown/PDF report

## Configuration

Edit [rcc_non_linear/config.py](rcc_non_linear/config.py) to customize:
- Default material properties
- Analysis parameters
- Output settings

## Contributing

This project is part of PhD research at University of Nevada, Reno. For contributions or issues, please contact the maintainers.

## License

See [LICENSE](LICENSE) file for details.

## References

- Mander, J.B., Priestley, M.J.N., Park, R. (1988). "Theoretical stress-strain model for confined concrete." Journal of Structural Engineering.
- OpenSees - Open System for Earthquake Engineering Simulation (https://opensees.berkeley.edu/)

## Contact

**Author**: Abinash Mandal  
**Affiliation**: University of Nevada, Reno - PhD in Structural Engineering
