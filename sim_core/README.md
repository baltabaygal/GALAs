# Axion Field Simulation Core v1.0

A high-performance, Numba-accelerated Python suite for simulating axion field dynamics during first-order phase transitions in the early universe.

## Physics Background

The core simulation solves the classical equation of motion for the axion field $\theta$ in an expanding Friedmann-Robertson-Walker (FRW) universe.

### 1. Physical Time ($t$)
In terms of cosmic time $t$, the evolution is governed by:
$$\ddot{\theta} + 3H\dot{\theta} - \frac{1}{a^2}\nabla^2\theta + V'(\theta) = 0$$
Where:
- $3H\dot{\theta}$ is the **Hubble friction** term.
- $\frac{1}{a^2}\nabla^2\theta$ is the **gradient (laplacian)** term, suppressed by the expansion.
- $V'(\theta) = m^2(t, \mathbf{x}) \sin\theta$ is the periodic potential.

### 2. Conformal Time ($\tau$)
For numerical stability and efficiency, the simulation is performed in **conformal time** ($d\tau = dt/a$). In radiation dominance ($a \propto \tau$), the equation becomes:
$$\theta'' + 2\mathcal{H}\theta' - \nabla^2\theta + a^2 V'(\theta) = 0$$
Where:
- $\theta'$ denotes the derivative with respect to conformal time ($\partial_\tau$).
- $\mathcal{H} = a'/a = 1/\tau$ is the conformal Hubble parameter.
- The gradient term $-\nabla^2\theta$ is calculated efficiently using spectral methods (FFT).


## Features

- **High Performance:** JIT-compiled kernels (via Numba) and spectral methods (via pyFFTW).
- **Optimized Nucleation:** Spatial hashing for efficient bubble-tracer interactions.
- **Cosmological Scaling:** Handles conformal time evolution and expansion effects.
- **Analysis Suite:** Automated data collapse and scaling analysis for research publication.

## Project Structure

- `sim_core/`
  - `axion_sim_v1p0.py`: The core simulation engine.
  - `run_sweep_v1p0.py`: Utility for running large parameter sweeps.
  - `run_single_v1p0.py`: Script for single runs and quick visualization.
  - `analyze_sweep_v1p0.py`: Data aggregation and physical scaling analysis.
- `run_single_v1p0.py`: Helper to run and plot a single test simulation.

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Environment Setup
```bash
conda create -n axion_sim python=3.9
conda activate axion_sim
conda install numpy matplotlib tqdm
conda install -c conda-forge numba pyfftw
```

## Usage

### 1. Run a Test Simulation
To verify the physics on a small grid (32x32) and see immediate results:
```bash
python3 -m sim_core.run_single_v1p0
```
This generates `single_sim_test.png` showing energy evolution.

### 2. Run a Parameter Sweep
To run a full sweep across different $H_{PT}$ and $\beta/H$ values:
```bash
python3 -m sim_core.run_sweep_v1p0
```
Results will be saved in `production_sweep_v1p0/data/`.

### 3. Analyze Results
To aggregate data and produce summary plots:
```bash
python3 -m sim_core.analyze_sweep_v1p0 production_sweep_v1p0
```
This will generate `energy_ratio_by_theta.png`, `data_collapse.png`, and more.

## Physics Parameters

- `H_PT`: The Hubble rate at the time of the phase transition.
- `beta/H`: The speed of the phase transition.
- `theta0`: The initial misalignment angle of the axion field.
- `v_bubble`: The wall velocity of the nucleated bubbles.

## Citation

If you use this code in your research, please cite:
[Your Paper/Reference Here]
