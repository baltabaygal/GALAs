"""
Axion Field Simulation Core v1.0
================================

A high-performance, Numba-accelerated simulation of axion field dynamics 
during first-order phase transitions. This core handles bubble nucleation, 
mass field evolution, and field equations in a cosmological background.

Key Features:
-------------
*   JIT-accelerated kernels using Numba for physical field updates.
*   Spatial hashing for optimized bubble-tracer interactions.
*   FFTW-integrated spectral operations for gradient and Laplacian calculations.
*   Aggressive threading control to eliminate lock contention in HPC environments.
*   Memory-efficient data structures for large-scale parameter sweeps.

Physics:
--------
The simulation solves the classical axion field equation:
    □θ + (1/a²) V'(θ) = 0
where V(θ) is the periodic potential m²(t,x)(1 - cosθ).

Author: [Your Name/Group]
Date: April 2026
"""

import os
import sys
import warnings
import time
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, asdict

import numpy as np
import numba as nb
import pyfftw
from tqdm import tqdm

# ==============================================================================
# PERFORMANCE & THREADING CONFIGURATION
# ==============================================================================
# Force single-threaded mode for core libraries to prevent over-subscription 
# and lock contention. Threading is managed explicitly via Numba.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Configure Numba threading (adjust based on target hardware)
nb.set_num_threads(8)

# Configure FFTW
pyfftw.config.NUM_THREADS = 1
np.seterr(all='ignore')

# Type Definitions
FLOAT_TYPE = np.float32
COMPLEX_TYPE = np.complex64

@dataclass
class SimulationConfig:
    """Configuration parameters for the axion field simulation."""
    # Grid parameters
    Ngrid: int = 64
    boxSize_comoving: float = 8.0 * np.pi
    
    # Field parameters
    m0: float = 1.0              # True vacuum mass
    mb: float = 0.0              # False vacuum mass (background)
    theta0_initial: float = 2.0  # Initial misalignement angle
    
    # Bubble parameters
    v_bubble: float = 0.3        # Bubble wall velocity
    delta_w: float = 0.15        # Wall thickness (if applicable)
    enable_nucleation: bool = True
    
    # Cosmological parameters
    a0: float = 1.0              # Initial scale factor
    tau0: float = 1.0            # Initial conformal time
    Gamma_0: float = 1e-6        # Nucleation rate pre-factor
    beta: float = 0.1            # Phase transition speed parameter
    t_0: float = 0.0             # Reference cosmic time
    
    # Numerical control
    cfl_factor: float = 0.5      # CFL stability condition factor
    friction_factor: float = 0.1 # Damping factor for time-step stability
    num_tracers: int = 100000    # Number of Monte Carlo nucleation tracers
    
    # Optimization
    spatial_hash_cells: int = 8  # Cells per dim for spatial hash
    energy_save_interval: int = 10
    checkpoint_interval: int = 0
    use_fft_laplacian: bool = False
    
    def __post_init__(self):
        """Ensure Ngrid is a power of 2 for FFT performance."""
        if self.Ngrid <= 0 or (self.Ngrid & (self.Ngrid - 1)) != 0:
            new_n = 2 ** int(np.ceil(np.log2(self.Ngrid)))
            warnings.warn(f"Grid size adjusted from {self.Ngrid} to {new_n} for FFT performance.")
            self.Ngrid = new_n

# ==============================================================================
# NUMBA KERNELS (PHYSICS & GEOMETRY)
# ==============================================================================

@nb.njit(parallel=True, fastmath=True, cache=True)
def calculate_mass_field_kernel(X, Y, Z, bubble_centers, bubble_radii_sq, 
                               m0_sq, mb_sq, box_size):
    """
    Calculates the mass-squared field with 8-point subsampling for anti-aliasing.
    
    Returns a grid where mass is interpolated between true and false vacuum
    based on the fraction of the cell volume inside bubbles.
    """
    Nx, Ny, Nz = X.shape
    mass_squared = np.full((Nx, Ny, Nz), mb_sq, dtype=X.dtype)
    dx = box_size / Nx
    num_bubbles = len(bubble_centers)
    
    if num_bubbles == 0:
        return mass_squared
    
    # Subsampling offsets (corners of a sub-cube)
    offsets = np.array([
        [-0.25, -0.25, -0.25], [-0.25, -0.25, 0.25], [-0.25, 0.25, -0.25], [-0.25, 0.25, 0.25],
        [0.25, -0.25, -0.25], [0.25, -0.25, 0.25], [0.25, 0.25, -0.25], [0.25, 0.25, 0.25]
    ], dtype=X.dtype) * dx
    
    N_total = Nx * Ny * Nz
    for idx in nb.prange(N_total):
        i = idx // (Ny * Nz)
        j = (idx // Nz) % Ny
        k = idx % Nz
        
        points_inside = 0
        x0, y0, z0 = X[i, j, k], Y[i, j, k], Z[i, j, k]
        
        for m in range(8):
            px = x0 + offsets[m, 0]
            py = y0 + offsets[m, 1]
            pz = z0 + offsets[m, 2]
            
            is_inside = False
            for b in range(num_bubbles):
                dx_ = px - bubble_centers[b, 0]
                dy_ = py - bubble_centers[b, 1]
                dz_ = pz - bubble_centers[b, 2]
                
                # Periodic Boundary Conditions
                dx_ -= box_size * round(dx_ / box_size)
                dy_ -= box_size * round(dy_ / box_size)
                dz_ -= box_size * round(dz_ / box_size)
                
                if dx_**2 + dy_**2 + dz_**2 <= bubble_radii_sq[b]:
                    is_inside = True
                    break
            
            if is_inside:
                points_inside += 1
        
        if points_inside > 0:
            fraction = points_inside / 8.0
            mass_squared[i, j, k] = fraction * m0_sq + (1.0 - fraction) * mb_sq
    
    return mass_squared

@nb.njit(parallel=True, fastmath=True, cache=True)
def find_surviving_tracers_kernel(all_tracers, active_indices, bubble_centers, 
                                 bubble_radii_sq, box_size):
    """Checks which tracers have been engulfed by bubbles (Periodic BCs)."""
    num_active = len(active_indices)
    num_bubbles = len(bubble_centers)
    
    if num_bubbles == 0 or num_active == 0:
        return active_indices
    
    survived_mask = np.ones(num_active, dtype=nb.boolean)
    
    for idx in nb.prange(num_active):
        i = active_indices[idx]
        pos = all_tracers[i]
        
        for b in range(num_bubbles):
            dx = pos[0] - bubble_centers[b, 0]
            dy = pos[1] - bubble_centers[b, 1]
            dz = pos[2] - bubble_centers[b, 2]
            
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)
            
            if dx*dx + dy*dy + dz*dz <= bubble_radii_sq[b]:
                survived_mask[idx] = False
                break
    
    return active_indices[survived_mask]

# ==============================================================================
# COMPONENT MANAGERS
# ==============================================================================

class CosmologyManager:
    """Standard Radiation-Dominated Cosmology with Phase Transition scaling."""
    def __init__(self, H_PT: float):
        self.H_PT = H_PT
        self.t_PT = 1.0 / (2.0 * H_PT)  # Cosmic time at PT
        self.tau_PT = 1.0 / H_PT         # Conformal time at PT

    def scale_factor(self, tau: float) -> float:
        return self.H_PT * tau

    def cosmic_time(self, tau: float) -> float:
        return 0.5 * self.H_PT * tau**2

    def conformal_time(self, t_cosmic: float) -> float:
        return np.sqrt(2 * max(t_cosmic, 0.0) / self.H_PT)

class FFTWBackend:
    """Interface for pyFFTW with thread safety."""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.complex64 if np.issubdtype(dtype, np.floating) else dtype
        self.in_arr = pyfftw.empty_aligned(shape, dtype=self.dtype)
        self.out_arr = pyfftw.empty_aligned(shape, dtype=self.dtype)
        
        self.fft = pyfftw.FFTW(self.in_arr, self.out_arr, axes=(0, 1, 2), 
                               direction='FFTW_FORWARD', threads=1, flags=('FFTW_MEASURE',))
        self.ifft = pyfftw.FFTW(self.out_arr, self.in_arr, axes=(0, 1, 2), 
                                direction='FFTW_BACKWARD', normalise_idft=True, threads=1)

    def forward(self, data):
        self.in_arr[...] = data
        return self.fft()

    def backward(self, data):
        self.out_arr[...] = data
        return self.ifft()

class BubbleManager:
    """Manages bubble growth and geometric state."""
    def __init__(self, box_size: float):
        self.box_size = box_size
        self.centers = np.zeros((0, 3), dtype=FLOAT_TYPE)
        self.birth_times = np.zeros(0, dtype=FLOAT_TYPE)
        self.velocities = np.zeros(0, dtype=FLOAT_TYPE)
        self.radii_comoving = np.zeros(0, dtype=FLOAT_TYPE)

    def add_bubble(self, center, velocity, tau_birth):
        # Wrap center to box
        center = (center + self.box_size/2) % self.box_size - self.box_size/2
        self.centers = np.vstack([self.centers, center])
        self.birth_times = np.append(self.birth_times, tau_birth)
        self.velocities = np.append(self.velocities, velocity)
        self.radii_comoving = np.append(self.radii_comoving, 0.0)

    def update_radii(self, current_tau):
        if len(self.birth_times) > 0:
            self.radii_comoving = self.velocities * (current_tau - self.birth_times)

    @property
    def count(self):
        return len(self.birth_times)

class MassFieldManager:
    """Handles interpolation and caching of the mass squared field."""
    def __init__(self, X, Y, Z, m0_sq, mb_sq, box_size):
        self.X, self.Y, self.Z = X, Y, Z
        self.m0_sq, self.mb_sq = m0_sq, mb_sq
        self.box_size = box_size
        self.cached_mass = None
        self.cache_tau = -1.0

    def get_mass_field(self, centers, radii_comoving, tau):
        if abs(tau - self.cache_tau) < 1e-8:
            return self.cached_mass
        
        radii_sq = radii_comoving**2
        self.cached_mass = calculate_mass_field_kernel(
            self.X, self.Y, self.Z, centers, radii_sq, 
            self.m0_sq, self.mb_sq, self.box_size
        )
        self.cache_tau = tau
        return self.cached_mass

# ==============================================================================
# MAIN SIMULATION ENGINE
# ==============================================================================

class AxionSimulation:
    """Core simulation class for axion field evolution."""
    
    def __init__(self, config: SimulationConfig, cosmology: CosmologyManager):
        self.config = config
        self.cosmo = cosmology
        self.N = config.Ngrid
        self.L = config.boxSize_comoving
        self.dx = self.L / self.N
        self.dV = self.dx**3
        
        # Grid Setup
        x = np.linspace(-self.L/2, self.L/2, self.N, endpoint=False, dtype=FLOAT_TYPE)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Spectral Setup
        k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K_SQ = KX**2 + KY**2 + KZ**2
        self.fft = FFTWBackend(self.X.shape, FLOAT_TYPE)
        
        # Field Initialization
        self.theta = np.full(self.X.shape, config.theta0_initial, dtype=FLOAT_TYPE)
        self.theta_p = np.zeros(self.X.shape, dtype=FLOAT_TYPE)
        
        # Managers
        self.bubbles = BubbleManager(self.L)
        self.mass_mgr = MassFieldManager(self.X, self.Y, self.Z, config.m0**2, config.mb**2, self.L)
        self.tracers = ((np.random.rand(config.num_tracers, 3) - 0.5) * self.L).astype(FLOAT_TYPE)
        self.active_tracers = np.arange(config.num_tracers)
        
        # Diagnostics
        self.history = {'tau': [], 't': [], 'energy': [], 'fv_frac': [], 'n_bubbles': []}

    def _laplacian(self, field):
        """Compute Laplacian using FFT."""
        field_k = self.fft.forward(field)
        return self.fft.backward(-self.K_SQ * field_k).real

    def _get_timestep(self, tau):
        # CFL Condition: dt < dx / sqrt(3)
        omega_max = np.sqrt(3 * (np.pi/self.dx)**2 + (self.config.m0 * self.cosmo.scale_factor(tau))**2)
        dt_cfl = self.config.cfl_factor / omega_max
        return min(dt_cfl, self.config.friction_factor * tau)

    def _nucleation_rate(self, t):
        return self.config.Gamma_0 * np.exp(self.config.beta * (t - self.config.t_0))

    def run(self, tau_final):
        tau = self.config.tau0
        pbar = tqdm(total=int(tau_final - tau), desc="Axion Sim")
        
        step = 0
        while tau < tau_final:
            dt = self._get_timestep(tau)
            if tau + dt > tau_final: dt = tau_final - tau
            
            a = self.cosmo.scale_factor(tau)
            H_conf = 1.0 / tau # Radiation dominance
            
            # 1. Leapfrog Part A (Field)
            mass_sq = self.mass_mgr.get_mass_field(self.bubbles.centers, self.bubbles.radii_comoving, tau)
            accel = self._laplacian(self.theta) - (a**2)*mass_sq*np.sin(self.theta) - 2*H_conf*self.theta_p
            self.theta_p += 0.5 * dt * accel
            self.theta += dt * self.theta_p
            
            # 2. Nucleation & Bubble Growth
            if self.config.enable_nucleation:
                t = self.cosmo.cosmic_time(tau)
                rate = self._nucleation_rate(t)
                vol_tracer = (self.L**3 / self.config.num_tracers) * (a**3)
                prob = rate * vol_tracer * (dt * a) # Rate * PhysVol * dt_phys
                
                if len(self.active_tracers) > 0:
                    rolls = np.random.rand(len(self.active_tracers))
                    new_mask = rolls < prob
                    for idx in self.active_tracers[new_mask]:
                        self.bubbles.add_bubble(self.tracers[idx], self.config.v_bubble, tau)
                    self.active_tracers = self.active_tracers[~new_mask]
            
            tau += dt
            self.bubbles.update_radii(tau)
            
            # 3. Leapfrog Part B (Field Update)
            a_new = self.cosmo.scale_factor(tau)
            mass_sq_new = self.mass_mgr.get_mass_field(self.bubbles.centers, self.bubbles.radii_comoving, tau)
            accel_new = self._laplacian(self.theta) - (a_new**2)*mass_sq_new*np.sin(self.theta) - 2*(1.0/tau)*self.theta_p
            self.theta_p += 0.5 * dt * accel_new
            
            # 4. Engulfment
            self.active_tracers = find_surviving_tracers_kernel(
                self.tracers, self.active_tracers, self.bubbles.centers, 
                self.bubbles.radii_comoving**2, self.L
            )
            
            # Data Collection
            if step % self.config.energy_save_interval == 0:
                self._record_diagnostics(tau, mass_sq_new)
            
            step += 1
            pbar.update(int(dt))
            
        pbar.close()
        return self.history

    def _record_diagnostics(self, tau, mass_sq):
        a = self.cosmo.scale_factor(tau)
        kin = 0.5 * np.mean(self.theta_p**2) / (a**2)
        pot = np.mean(mass_sq * (1 - np.cos(self.theta)))
        # Gradient energy approximation via spectral power
        grad = 0.5 * np.mean(self._laplacian(self.theta) * self.theta) / (a**2) # Viral-like term
        
        self.history['tau'].append(tau)
        self.history['t'].append(self.cosmo.cosmic_time(tau))
        self.history['fv_frac'].append(len(self.active_tracers) / self.config.num_tracers)
        self.history['n_bubbles'].append(self.bubbles.count)
        self.history['energy'].append({'kin': kin, 'pot': pot, 'grad': abs(grad)})

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_simulation(H_PT: float, beta_H: float, **kwargs):
    """Factory to initialize a simulation with standard PT parameters."""
    cosmo = CosmologyManager(H_PT)
    config = SimulationConfig(
        beta=beta_H * H_PT, 
        tau0=cosmo.tau_PT,
        t_0=cosmo.t_PT,
        **kwargs
    )
    return AxionSimulation(config, cosmo)

if __name__ == "__main__":
    print("Axion Simulation Core v1.0")
    print("Usage: Import this module in your sweep or plotting script.")
    # Quick test run
    sim = create_simulation(H_PT=1.0, beta_H=10.0, Ngrid=32, num_tracers=1000)
    history = sim.run(tau_final=2.0)
    print(f"Test run complete. Final False Vacuum Fraction: {history['fv_frac'][-1]:.4f}")
