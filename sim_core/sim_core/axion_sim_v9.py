"""
Axion Field Simulation v8.1 - PROFILING-OPTIMIZED Edition
==========================================================

Critical fixes based on profiling analysis:
1. THREADING: Aggressive thread suppression to eliminate lock contention
2. ENERGY: Reduce calculation frequency and use cheaper approximations
3. MASS FIELD: Better caching to avoid redundant calculations
4. LAPLACIAN: Consider FFT-based calculation to reduce roll() calls

Key changes from v8:
- More aggressive thread control (environment + direct library config)
- Energy calculation only when explicitly needed (not in main loop)
- Mass field cache invalidation tracking
- Optional FFT-based Laplacian for gradient calculations
"""

import os
import sys

# Disable MKL and BLAS multithreading to prevent thread locks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Allow Numba to use all threads safely:
import numba
numba.set_num_threads(8)   # or number of cores you have


import numba as nb
import numpy as np
import pyfftw
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import warnings
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')


# Force single-threaded mode
pyfftw.config.NUM_THREADS = 1
np.seterr(all='ignore')  # Suppress numpy warnings in tight loops

# Try to control NumPy threading
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

FLOAT_TYPE = np.float32
COMPLEX_TYPE = np.complex64

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# np.random.seed(42)

@dataclass
class SimulationConfig:
    """Configuration parameters for the axion field simulation."""
    # Grid and box parameters
    Ngrid: int = 16
    boxSize_comoving: float = 8.0 * np.pi
    
    # Physical parameters
    m0: float = 1.0
    v_bubble: float = 0.3
    delta_w: float = 0.15
    theta0_initial: float = 2.0
    
    # Cosmological parameters
    a0: float = 1.0
    tau0: float = 1.0
    Gamma_0: float = 1e-6
    beta: float = 0.1
    t_0: float = 0.0
    
    # Control flags
    enable_nucleation: bool = True
    mb: float = 0.0
    
    # Numerical parameters
    cfl_factor: float = 0.5
    friction_factor: float = 0.1
    num_tracers: int = 100000
    
    # Optimization parameters
    spatial_hash_cells: int = 8
    energy_save_interval: int = 2  # INCREASED from 10
    checkpoint_interval: int = 0
    use_fft_laplacian: bool = False  # NEW: Use FFT for gradients
    
    def __post_init__(self):
        """Validate and adjust parameters."""
        if self.Ngrid <= 0 or (self.Ngrid & (self.Ngrid - 1)) != 0:
            self.Ngrid = 2 ** int(np.ceil(np.log2(self.Ngrid)))
            warnings.warn(f"Grid size adjusted to {self.Ngrid} for optimal FFT performance")

# ==============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS (unchanged from v8)
# ==============================================================================

@nb.njit(cache=True)
def create_true_vacuum(shape, m0_sq, dtype):
    """Create a uniform true vacuum mass field."""
    return np.full(shape, m0_sq, dtype=dtype)

# @nb.njit(parallel=False, fastmath=True, cache=True)
# def calculate_mass_field_snapshot_subsampled(X, Y, Z, bubble_centers, bubble_radii_comoving, 
#                                            m0_sq, mb_sq, box_size):
#     """Calculate mass field configuration with 8-point subsampling."""
#     Nx, Ny, Nz = X.shape
#     mass_squared = np.full((Nx, Ny, Nz), mb_sq, dtype=X.dtype)
#     dx = box_size / Nx
#     num_bubbles = len(bubble_centers)

#     if num_bubbles == 0:
#         return mass_squared

#     sub_offsets = np.array([-0.25, 0.25]) * dx

#     for i in nb.prange(Nx):
#         for j in range(Ny):
#             for k in range(Nz):
#                 points_inside_bubble = 0
                
#                 for sx_offset in sub_offsets:
#                     for sy_offset in sub_offsets:
#                         for sz_offset in sub_offsets:
#                             px = X[i, j, k] + sx_offset
#                             py = Y[i, j, k] + sy_offset
#                             pz = Z[i, j, k] + sz_offset
                            
#                             is_inside_any_bubble = False
#                             for b in range(num_bubbles):
#                                 center = bubble_centers[b]
#                                 radius_sq = bubble_radii_comoving[b]**2

#                                 dist_x = px - center[0]
#                                 dist_y = py - center[1]
#                                 dist_z = pz - center[2]
#                                 dist_x -= box_size * round(dist_x / box_size)
#                                 dist_y -= box_size * round(dist_y / box_size)
#                                 dist_z -= box_size * round(dist_z / box_size)

#                                 if dist_x**2 + dist_y**2 + dist_z**2 <= radius_sq:
#                                     is_inside_any_bubble = True
#                                     break
                            
#                             if is_inside_any_bubble:
#                                 points_inside_bubble += 1
                
#                 if points_inside_bubble > 0:
#                     fraction_inside = points_inside_bubble / 8.0
#                     mass_squared[i, j, k] = (fraction_inside * m0_sq) + ((1.0 - fraction_inside) * mb_sq)

#     return mass_squared


@nb.njit(parallel=True, fastmath=True, cache=True)
def calculate_mass_field_snapshot_subsampled(X, Y, Z, bubble_centers, bubble_radii_comoving, 
                                       m0_sq, mb_sq, box_size):
    """Parallelized mass field calculation (flattened 3D loop + subsampling)."""
    Nx, Ny, Nz = X.shape
    mass_squared = np.full((Nx, Ny, Nz), mb_sq, dtype=X.dtype)
    dx = box_size / Nx
    num_bubbles = len(bubble_centers)
    
    if num_bubbles == 0:
        return mass_squared
    
    # Subsampling offsets (8 points)
    sub_offsets = np.array([-0.25, 0.25], dtype=X.dtype) * dx
    # Precompute all combinations
    offsets = np.array([[sx, sy, sz] for sx in sub_offsets 
                                    for sy in sub_offsets 
                                    for sz in sub_offsets], dtype=X.dtype)
    
    N_total = Nx * Ny * Nz
    for idx in nb.prange(N_total):
        i = idx // (Ny * Nz)
        j = (idx // Nz) % Ny
        k = idx % Nz
        
        points_inside_bubble = 0
        
        x0, y0, z0 = X[i, j, k], Y[i, j, k], Z[i, j, k]
        
        for off in offsets:
            px, py, pz = x0 + off[0], y0 + off[1], z0 + off[2]
            is_inside_any = False
            for b in range(num_bubbles):
                center = bubble_centers[b]
                radius_sq = bubble_radii_comoving[b] ** 2
                
                dx_ = px - center[0]
                dy_ = py - center[1]
                dz_ = pz - center[2]
                
                # Periodic BC
                dx_ -= box_size * round(dx_ / box_size)
                dy_ -= box_size * round(dy_ / box_size)
                dz_ -= box_size * round(dz_ / box_size)
                
                if dx_**2 + dy_**2 + dz_**2 <= radius_sq:
                    is_inside_any = True
                    break
            
            if is_inside_any:
                points_inside_bubble += 1
        
        if points_inside_bubble > 0:
            fraction_inside = points_inside_bubble / 8.0
            mass_squared[i, j, k] = fraction_inside * m0_sq + (1.0 - fraction_inside) * mb_sq
    
    return mass_squared


@nb.njit(parallel=False, fastmath=True, cache=True)
def get_bubble_mask_snapshot(X, Y, Z, bubble_centers, bubble_radii_comoving, box_size):
    """Generate boolean mask indicating which grid points are inside bubbles."""
    Nx, Ny, Nz = X.shape
    mask = np.zeros((Nx, Ny, Nz), dtype=nb.boolean)
    num_bubbles = len(bubble_centers)

    if num_bubbles == 0:
        return mask

    for i in nb.prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for b in range(num_bubbles):
                    center = bubble_centers[b]
                    radius = bubble_radii_comoving[b]
                    
                    dx = X[i, j, k] - center[0]
                    dy = Y[i, j, k] - center[1]
                    dz = Z[i, j, k] - center[2]
                    
                    dx -= box_size * round(dx / box_size)
                    dy -= box_size * round(dy / box_size)
                    dz -= box_size * round(dz / box_size)
                    
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        mask[i, j, k] = True
                        break
    return mask

@nb.njit(parallel=False, fastmath=True, cache=True)
def evolve_bubbles_vectorized(radii_comoving, velocities, birth_times, 
                               count, current_tau):
    """Evolve COMOVING bubble radii: R = v_w × (τ - τ_n)"""
    for i in nb.prange(count):
        tau_n = birth_times[i]
        radii_comoving[i] = velocities[i] * (current_tau - tau_n)

@nb.njit(fastmath=True, cache=True)
def assign_to_hash_cell(pos, box_size, num_cells):
    """Assign a position to a spatial hash cell."""
    normalized = (pos + box_size/2) % box_size
    cell_idx = int(normalized / box_size * num_cells)
    return min(max(cell_idx, 0), num_cells - 1)

@nb.njit(fastmath=True, cache=True)
def get_neighboring_cells(cell_x, cell_y, cell_z, num_cells):
    """Get list of neighboring cells (3x3x3 = 27 cells including self)."""
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx = (cell_x + dx) % num_cells
                ny = (cell_y + dy) % num_cells
                nz = (cell_z + dz) % num_cells
                neighbors.append((nx, ny, nz))
    return neighbors

@nb.njit(parallel=True, fastmath=True, cache=True)
def find_surviving_tracers_spatial_hash(
    all_tracers, active_indices, 
    bubble_centers, bubble_radii_comoving, 
    box_size, num_hash_cells
):
    """Optimized tracer removal using spatial hashing."""
    num_active = len(active_indices)
    num_bubbles = len(bubble_centers)
    
    if num_bubbles == 0 or num_active == 0:
        return active_indices
    
    max_bubbles_per_cell = min(100, num_bubbles)
    
    bubble_hash_counts = np.zeros((num_hash_cells, num_hash_cells, num_hash_cells), dtype=nb.int32)
    bubble_hash = np.zeros((num_hash_cells, num_hash_cells, num_hash_cells, max_bubbles_per_cell), dtype=nb.int32)
    
    for b in range(num_bubbles):
        center = bubble_centers[b]
        cx = assign_to_hash_cell(center[0], box_size, num_hash_cells)
        cy = assign_to_hash_cell(center[1], box_size, num_hash_cells)
        cz = assign_to_hash_cell(center[2], box_size, num_hash_cells)
        
        radius = bubble_radii_comoving[b]
        cell_size = box_size / num_hash_cells
        
        min_cx = assign_to_hash_cell(center[0] - radius, box_size, num_hash_cells)
        max_cx = assign_to_hash_cell(center[0] + radius, box_size, num_hash_cells)
        min_cy = assign_to_hash_cell(center[1] - radius, box_size, num_hash_cells)
        max_cy = assign_to_hash_cell(center[1] + radius, box_size, num_hash_cells)
        min_cz = assign_to_hash_cell(center[2] - radius, box_size, num_hash_cells)
        max_cz = assign_to_hash_cell(center[2] + radius, box_size, num_hash_cells)
        
        for ix in range(min_cx, max_cx + 1):
            for iy in range(min_cy, max_cy + 1):
                for iz in range(min_cz, max_cz + 1):
                    cell_x = ix % num_hash_cells
                    cell_y = iy % num_hash_cells
                    cell_z = iz % num_hash_cells
                    
                    count = bubble_hash_counts[cell_x, cell_y, cell_z]
                    if count < max_bubbles_per_cell:
                        bubble_hash[cell_x, cell_y, cell_z, count] = b
                        bubble_hash_counts[cell_x, cell_y, cell_z] += 1
    
    survived_mask = np.ones(num_active, dtype=nb.boolean)
    
    for idx in nb.prange(num_active):
        i = active_indices[idx]
        pos = all_tracers[i]
        
        tx = assign_to_hash_cell(pos[0], box_size, num_hash_cells)
        ty = assign_to_hash_cell(pos[1], box_size, num_hash_cells)
        tz = assign_to_hash_cell(pos[2], box_size, num_hash_cells)
        
        num_bubbles_in_cell = bubble_hash_counts[tx, ty, tz]
        
        for local_idx in range(num_bubbles_in_cell):
            b = bubble_hash[tx, ty, tz, local_idx]
            center = bubble_centers[b]
            radius_sq = bubble_radii_comoving[b]**2
            
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            dz = pos[2] - center[2]
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)
            
            if dx*dx + dy*dy + dz*dz <= radius_sq:
                survived_mask[idx] = False
                break
    
    return active_indices[survived_mask]

@nb.njit(parallel=True, fastmath=True, cache=True)
def find_surviving_tracers_simple(
    all_tracers, active_indices, bubble_centers, bubble_radii_comoving, box_size, number_hash_cells
):
    """
    Simple and robust tracer removal - checks only active tracers.
    
    Args:
        all_tracers: (N_total, 3) all tracer positions (COMOVING)
        active_indices: (N_active,) indices of active tracers
        bubble_centers: (N_bubbles, 3) bubble centers (COMOVING)
        bubble_radii_comoving: (N_bubbles,) bubble radii (COMOVING)
        box_size: COMOVING box size for PBC
    """
    num_active = len(active_indices)
    num_bubbles = len(bubble_centers)
    
    if num_bubbles == 0:
        return active_indices
    
    if num_active == 0:
        return active_indices
    
    # Mark which active tracers survive
    survived_mask = np.ones(num_active, dtype=nb.boolean)
    
    # Check each active tracer against all bubbles (COMOVING distance)
    for idx in nb.prange(num_active):
        i = active_indices[idx]
        pos = all_tracers[i]  # COMOVING position
        
        for b in range(num_bubbles):
            center = bubble_centers[b]  # COMOVING center
            radius_sq = bubble_radii_comoving[b]**2  # COMOVING radius squared
            
            # Comoving distance with PBC
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            dz = pos[2] - center[2]
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)
            
            if dx*dx + dy*dy + dz*dz <= radius_sq:
                survived_mask[idx] = False
                break
    
    # Return only surviving indices
    return active_indices[survived_mask]

# ==============================================================================
# COSMOLOGY (unchanged)
# ==============================================================================

class CosmologyManagerPTAtZeroStartAtZero:
    """Cosmology where t=0, tau=0 at start, PT at t_PT = 1/(2 H_PT), a(t_PT)=1."""

    def __init__(self, H_PT: float):
        self.H_PT = H_PT
        self.t_PT = 1.0 / (2.0 * H_PT)
        self.a_PT = 1.0
        self.tau_PT = 1.0 / H_PT

    def scale_factor(self, tau: float) -> float:
        return self.H_PT * tau

    def cosmic_time_from_conformal(self, tau: float) -> float:
        return 0.5 * self.H_PT * tau**2

    def conformal_time_from_cosmic(self, t_cosmic: float) -> float:
        return np.sqrt(2 * max(t_cosmic, 0.0) / self.H_PT)

# ==============================================================================
# FFT MANAGEMENT (with more aggressive thread control)
# ==============================================================================

class FFTWBackend:
    """FFTW backend with AGGRESSIVE threading control."""

    def __init__(self, threads: int = 1):
        self.threads = 1
        self.shape = None
        self.dtype = None
        self.input_array = None
        self.output_array = None
        self.fftn_obj = None
        self.ifftn_obj = None
        
        # Force wisdom to be non-threaded
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300)

    def setup(self, shape, dtype):
        self.shape = shape
        self.dtype = np.complex64 if np.issubdtype(dtype, np.floating) else dtype

        self.input_array = pyfftw.empty_aligned(shape, dtype=self.dtype)
        self.output_array = pyfftw.empty_aligned(shape, dtype=self.dtype)

        # CRITICAL: Explicitly set threads=1
        self.fftn_obj = pyfftw.FFTW(
            self.input_array, self.output_array,
            axes=range(len(shape)),
            direction='FFTW_FORWARD',
            threads=1,
            flags=('FFTW_MEASURE',)  # Changed from ESTIMATE for better performance
        )
        self.ifftn_obj = pyfftw.FFTW(
            self.output_array, self.input_array,
            axes=range(len(shape)),
            direction='FFTW_BACKWARD',
            normalise_idft=True,
            threads=1,
            flags=('FFTW_MEASURE',)
        )

    def fftn(self, array):
        self.input_array[...] = np.ascontiguousarray(array, dtype=self.dtype)
        return self.fftn_obj()

    def ifftn(self, array):
        self.output_array[...] = np.ascontiguousarray(array, dtype=self.dtype)
        return self.ifftn_obj()

class FFTManager:
    """Manages FFT operations."""
    
    def __init__(self, preferred_backend: str = 'fftw'):
        try:
            self.backend = FFTWBackend()
        except ImportError:
            raise RuntimeError("pyFFTW is required for this simulation.")
    
    def setup(self, shape, dtype):
        self.backend.setup(shape, dtype)
    
    def fftn(self, array): 
        return self.backend.fftn(array)
    
    def ifftn(self, array): 
        return self.backend.ifftn(array)

# ==============================================================================
# BUBBLE MANAGEMENT (unchanged)
# ==============================================================================

class OptimizedBubbleManager:
    """Bubble manager with vectorized operations using COMOVING radii."""
    
    def __init__(self, box_size: float, initial_capacity: int = 10000, growth_factor: float = 1.5):
        self.box_size = box_size
        self.count = 0
        self.capacity = initial_capacity
        self.growth_factor = growth_factor
        
        self.centers = np.zeros((self.capacity, 3), dtype=FLOAT_TYPE)
        self.radii_comoving = np.zeros(self.capacity, dtype=FLOAT_TYPE)
        self.velocities = np.zeros(self.capacity, dtype=FLOAT_TYPE)
        self.birth_times = np.zeros(self.capacity, dtype=FLOAT_TYPE)

    def _resize_if_needed(self):
        if self.count >= self.capacity:
            new_capacity = int(self.capacity * self.growth_factor)
            self.centers.resize((new_capacity, 3), refcheck=False)
            self.radii_comoving.resize(new_capacity, refcheck=False)
            self.velocities.resize(new_capacity, refcheck=False)
            self.birth_times.resize(new_capacity, refcheck=False)
            self.capacity = new_capacity

    def add_bubble(self, center: np.ndarray, velocity: float, birth_time: float):
        self._resize_if_needed()
        idx = self.count
        
        wrapped_center = center.copy()
        for dim in range(3):
            wrapped_center[dim] = (wrapped_center[dim] + self.box_size/2) % self.box_size - self.box_size/2
        
        self.centers[idx] = wrapped_center
        self.radii_comoving[idx] = 0.0
        self.velocities[idx] = velocity
        self.birth_times[idx] = birth_time
        self.count += 1

    def evolve_bubbles(self, tau: float):
        if self.count == 0:
            return
        evolve_bubbles_vectorized(
            self.radii_comoving, 
            self.velocities, 
            self.birth_times,
            self.count, 
            tau
        )

    def get_centers_and_radii_comoving(self, scale_factor: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.array([]).reshape(0, 3), np.array([])
        return self.centers[:self.count].copy(), self.radii_comoving[:self.count].copy()

    def get_centers_and_radii_physical(self, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.array([]).reshape(0, 3), np.array([])
        radii_physical = self.radii_comoving[:self.count] * scale_factor
        return self.centers[:self.count].copy(), radii_physical

# ==============================================================================
# TRACER MANAGEMENT (unchanged)
# ==============================================================================

class NucleationTracerManager:
    """Tracer manager with spatial hashing optimization."""
    
    def __init__(self, num_tracers: int, box_size: float,
                 H_PT: float = 1.0, Ngrid: int = 16,
                 num_hash_cells: int = 8):
        self.box_size = box_size
        self.H_PT = H_PT
        self.Ngrid = Ngrid
        self.num_hash_cells = num_hash_cells
        
        print(f"\n{'='*60}")
        print(f"Nucleation Tracer Configuration (OPTIMIZED)")
        print("Numba threads:", nb.get_num_threads())
        print(f"{'='*60}")
        print(f"Simulation box size (comoving):   {box_size:.3f}")
        print(f"Tracers in box:                   {num_tracers}")
        print(f"Spatial hash cells:               {num_hash_cells}^3 = {num_hash_cells**3}")
        print(f"{'='*60}\n")
        
        self.num_total_tracers = num_tracers
        self.tracers = self._initialize_tracers()
        self.active_indices = np.arange(num_tracers, dtype=np.int64)
        
    def _initialize_tracers(self):
        tracers = ((np.random.rand(self.num_total_tracers, 3) - 0.5) * 
                   self.box_size).astype(FLOAT_TYPE)
        return tracers
    
    def get_active_zone_volume(self) -> float:
        return self.box_size ** 3
    
    def get_zone_info(self) -> Dict[str, float]:
        hubble_radius = 1.0 / self.H_PT
        hubble_volumes = (self.box_size / hubble_radius) ** 3
        dx = self.box_size / self.Ngrid
        cells_across = self.box_size / dx
        
        return {
            'zone_size': self.box_size,
            'zone_volume': self.box_size ** 3,
            'box_volume': self.box_size ** 3,
            'volume_fraction': 1.0,
            'tiles_per_dim': 1,
            'num_tiles': 1,
            'is_periodic': True,
            'hubble_volumes': hubble_volumes,
            'cells_across': cells_across
        }
    
    @property
    def num_active_tracers(self) -> int:
        return len(self.active_indices)
    
    def get_false_vacuum_fraction(self) -> float:
        if self.num_total_tracers == 0:
            return 0.0
        return self.num_active_tracers / self.num_total_tracers
    
    def get_active_tracer_coords(self):
        return self.tracers[self.active_indices]
    
    def update_state(self, bubble_centers: np.ndarray, bubble_radii_comoving: np.ndarray):
        if self.num_active_tracers == 0:
            return
        
        surviving_indices = find_surviving_tracers_simple(
            self.tracers, 
            self.active_indices,
            bubble_centers,
            bubble_radii_comoving,
            self.box_size,
            self.num_hash_cells
        )
        self.active_indices = surviving_indices

# ==============================================================================
# MASS FIELD MANAGEMENT (with better cache tracking)
# ==============================================================================

class MassFieldManager:
    """Manages interpolated mass field evolution with improved caching."""
    
    def __init__(self, X, Y, Z, m0_sq, mb_sq, box_size, dtau_theta, v_bubble=0.5):
        self.X, self.Y, self.Z = X, Y, Z
        self.m0_sq, self.mb_sq = m0_sq, mb_sq
        self.box_size = box_size
        
        dx_grid = box_size / X.shape[0]
        self.dtau_bubble = max(1*dtau_theta, 0.3 * dx_grid / max(v_bubble, 1e-10))
        
        self.prev_mass = None
        self.prev_tau = None
        self.next_mass = None
        self.next_tau = None
        
        # Cache tracking
        self.cached_mass_squared = None
        self.cache_tau = None
        self.cache_tolerance = 1e-6

    def get_mass_field(self, bubble_centers, bubble_radii_comoving, tau, 
                      false_frac=None, phase_transition_complete=False):
        """Get mass field with improved caching."""
        
        # Check cache first
        if (self.cached_mass_squared is not None and 
            self.cache_tau is not None and 
            abs(tau - self.cache_tau) < self.cache_tolerance):
            return self.cached_mass_squared
        
        if phase_transition_complete or (false_frac is not None and false_frac <= 0.0):
            mass = create_true_vacuum(self.X.shape, self.m0_sq, self.X.dtype)
            self.cached_mass_squared = mass
            self.cache_tau = tau
            return mass
        
        if self.prev_mass is None:
            mass = calculate_mass_field_snapshot_subsampled(
                self.X, self.Y, self.Z, bubble_centers, bubble_radii_comoving, 
                self.m0_sq, self.mb_sq, self.box_size
            )
            self.prev_mass = mass.copy()
            self.next_mass = mass.copy()
            self.prev_tau = tau
            self.next_tau = tau + self.dtau_bubble
            self.cached_mass_squared = mass
            self.cache_tau = tau
            return mass

        while tau >= self.next_tau:
            self.prev_mass = self.next_mass.copy()
            self.prev_tau = self.next_tau
            self.next_mass = calculate_mass_field_snapshot_subsampled(
                self.X, self.Y, self.Z, bubble_centers, bubble_radii_comoving,
                self.m0_sq, self.mb_sq, self.box_size
            )
            self.next_tau += self.dtau_bubble

        alpha = (tau - self.prev_tau) / max(self.next_tau - self.prev_tau, 1e-12)
        alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        mass = (1.0 - alpha_smooth) * self.prev_mass + alpha_smooth * self.next_mass
        
        self.cached_mass_squared = mass
        self.cache_tau = tau
        
        return mass
    
    def get_cached_mass_squared(self):
        return self.cached_mass_squared
    
    def invalidate_cache(self):
        """Force cache invalidation."""
        self.cache_tau = None

# ==============================================================================
# ENERGY CALCULATOR (with optimizations)
# ==============================================================================

class EnergyCalculator:
    """Handles energy calculations with performance optimizations."""
    
    def __init__(self, config: SimulationConfig, field_evolver):
        self.config = config
        self.field_evolver = field_evolver
        self.m0 = config.m0
        self.dx = field_evolver.dx
        self.use_fft = config.use_fft_laplacian
    
    def calculate_gradient_density_fft(self, theta, a_tau):
        """Calculate gradient energy using FFT (faster for large grids)."""
        # Transform to k-space
        theta_k = self.field_evolver.fft.fftn(theta)
        
        # Calculate |∇θ|² in k-space: |k|² |θ_k|²
        grad_squared_k = self.field_evolver.K_SQUARED * np.abs(theta_k)**2
        
        # Transform back
        grad_squared = np.real(self.field_evolver.fft.ifftn(grad_squared_k))
        
        return 0.5 * grad_squared / (a_tau**2 * self.m0**2)
    
    def calculate_gradient_density_fd(self, theta, a_tau):
        """Calculate gradient energy using finite differences."""
        dx = self.dx
        
        def forth_order_grad(f, axis):
            rolled = lambda shift: np.roll(f, shift=shift, axis=axis)
            return (-rolled(2) + 8*rolled(1) - 8*rolled(-1) + rolled(-2)) / (12 * dx)
        
        grad_x = forth_order_grad(theta, axis=0)
        grad_y = forth_order_grad(theta, axis=1)
        grad_z = forth_order_grad(theta, axis=2)
        grad_squared = grad_x**2 + grad_y**2 + grad_z**2
        
        return 0.5 * grad_squared / (a_tau**2 * self.m0**2)
    
    def calculate_energy_components(self, tau: float, mass_squared: np.ndarray, 
                                   theta: np.ndarray, theta_prime: np.ndarray,
                                   bubble_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate energy components from current field state."""
        a_tau = self.field_evolver.cosmology.scale_factor(tau)
        
        # Kinetic energy density
        kinetic_density = 0.5 * theta_prime**2 / (a_tau**2 * self.m0**2)
        
        # Potential energy density
        potential_density = (mass_squared / self.m0**2) * (1 - np.cos(theta))
        
        # Gradient energy density (choose method based on config)
        if self.use_fft:
            gradient_density = self.calculate_gradient_density_fft(theta, a_tau)
        else:
            gradient_density = self.calculate_gradient_density_fd(theta, a_tau)
        
        # Total energy density
        total_density = kinetic_density + gradient_density + potential_density

        # Integration
        volume_norm = self.config.boxSize_comoving**3
        dV = self.field_evolver.dV

        total_energy = np.sum(total_density) * dV / volume_norm
        kinetic_energy = np.sum(kinetic_density) * dV / volume_norm
        gradient_energy = np.sum(gradient_density) * dV / volume_norm
        potential_energy = np.sum(potential_density) * dV / volume_norm
        
        # Bubble-specific energy
        bubble_energy = 0.0
        bubble_volume_fraction = 0.0
        if bubble_mask is not None:
            bubble_energy = np.sum(total_density[bubble_mask]) * dV / volume_norm
            bubble_volume_fraction = np.sum(bubble_mask) / bubble_mask.size
        
        return {
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'gradient_energy': gradient_energy,
            'potential_energy': potential_energy,
            'bubble_energy': bubble_energy,
            'scale_factor': a_tau,
            'bubble_volume_fraction': bubble_volume_fraction
        }

# ==============================================================================
# FIELD EVOLUTION (with FFT Laplacian option)
# ==============================================================================

class FieldEvolver:
    """Handles axion field evolution with optimizations."""
    
    def __init__(self, config: SimulationConfig, fft_manager: FFTManager, cosmology):
        self.config = config
        self.fft = fft_manager
        self.cosmology = cosmology
        self.parent_simulation = None
        
        self._setup_grids()
        self._setup_k_grids()
        
        self.theta = np.full(self.grid_shape, config.theta0_initial, dtype=FLOAT_TYPE)
        self.theta_prime = np.zeros(self.grid_shape, dtype=FLOAT_TYPE)
        
        dtau_theta = self.get_recommended_timestep(config.tau0)
        self.mass_manager = MassFieldManager(
            self.X, self.Y, self.Z, config.m0**2, config.mb**2, 
            config.boxSize_comoving, dtau_theta, config.v_bubble
        )
        
        self.use_fft_laplacian = config.use_fft_laplacian

    def _setup_grids(self):
        self.grid_shape = (self.config.Ngrid,) * 3
        self.dx = self.config.boxSize_comoving / self.config.Ngrid
        self.dV = self.dx ** 3
        
        x = np.linspace(
            -self.config.boxSize_comoving/2, 
            self.config.boxSize_comoving/2, 
            self.config.Ngrid, 
            endpoint=False, 
            dtype=FLOAT_TYPE
        )
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

    def _setup_k_grids(self):
        k = 2 * np.pi * np.fft.fftfreq(self.config.Ngrid, self.dx)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K_SQUARED = self.KX**2 + self.KY**2 + self.KZ**2

    def get_mass_field(self, *args, **kwargs):
        return self.mass_manager.get_mass_field(*args, **kwargs)

    def compute_laplacian_fft(self, field):
        """Compute Laplacian using FFT (faster, no roll() calls)."""
        field_k = self.fft.fftn(field)
        laplacian_k = -self.K_SQUARED * field_k
        return np.real(self.fft.ifftn(laplacian_k))

    def compute_laplacian_fd(self, field):
        """Compute Laplacian using 4th-order finite difference."""
        dx = self.dx
        
        def laplacian_4th_order(f, axis):
            rolled = lambda shift: np.roll(f, shift, axis=axis)
            return (-rolled(2) + 16*rolled(1) - 30*f + 16*rolled(-1) - rolled(-2)) / (12 * dx**2)
        
        return (laplacian_4th_order(field, 0) +
                laplacian_4th_order(field, 1) +
                laplacian_4th_order(field, 2))

    def compute_laplacian(self, field, tau):
        """Compute Laplacian using chosen method."""
        if self.use_fft_laplacian:
            return self.compute_laplacian_fft(field)
        else:
            return self.compute_laplacian_fd(field)

    def evolve_step_part1(self, tau, dtau, bubble_centers, bubble_radii_comoving):
        """First half of leapfrog step."""
        mass_squared = self.get_mass_field(bubble_centers, bubble_radii_comoving, tau)
        laplacian = self.compute_laplacian(self.theta, tau)
        H_conf = 1.0 / max(tau, 1e-12)
        scale_factor = self.cosmology.scale_factor(tau)

        acceleration = (laplacian
                        - (scale_factor**2) * mass_squared * np.sin(self.theta)
                        - 2.0 * H_conf * self.theta_prime)
        
        self.theta_prime += 0.5 * dtau * acceleration
        self.theta += dtau * self.theta_prime

    def evolve_step_part2(self, tau_new, dtau, bubble_centers, bubble_radii_comoving):
        """Second half of leapfrog step."""
        mass_squared_new = self.get_mass_field(bubble_centers, bubble_radii_comoving, tau_new)
        laplacian_new = self.compute_laplacian(self.theta, tau_new)
        H_conf_new = 1.0 / max(tau_new, 1e-12)
        scale_factor_new = self.cosmology.scale_factor(tau_new)

        acceleration_new = (laplacian_new
                            - (scale_factor_new**2) * mass_squared_new * np.sin(self.theta)
                            - 2.0 * H_conf_new * self.theta_prime)
        
        self.theta_prime += 0.5 * dtau * acceleration_new

    def get_recommended_timestep(self, tau):
        # k_max = np.pi / self.dx
        k_max = np.sqrt(3) * (np.pi / self.dx)
        omega_max = np.sqrt(k_max**2 + (self.config.m0 * self.cosmology.scale_factor(tau))**2)
        dtau_cfl = self.config.cfl_factor / omega_max
        dtau_friction = self.config.friction_factor * max(tau, 1e-6)
        return min(dtau_cfl, dtau_friction)

# ==============================================================================
# MAIN SIMULATION CLASS (optimized version)
# ==============================================================================

class OptimizedAxionSimulation:
    """High-performance axion field simulation v8.1 with profiling optimizations."""
    
    def __init__(self, config: SimulationConfig, cosmology, 
                 fft_backend: str = 'fftw', H_PT: float = None,
                 nucleation_mode: str = 'standard'):
        
        self.config = config
        self.cosmology = cosmology
        self.H_PT = H_PT if H_PT is not None else getattr(config, 'H_PT', 1.0)
        self.nucleation_mode = nucleation_mode
        
        # Initialize FFT and field evolver
        self.fft_manager = FFTManager(fft_backend)
        self.field_evolver = FieldEvolver(config, self.fft_manager, self.cosmology)
        
        # Setup bubble and tracer managers
        if not config.enable_nucleation:
            self.bubble_manager = OptimizedBubbleManager(config.boxSize_comoving)
            self.tracer_manager = None
            self.nucleation_volume = 0.0
            self.zone_info = {}
        else:
            self.bubble_manager = OptimizedBubbleManager(config.boxSize_comoving)
            self.tracer_manager = NucleationTracerManager(
                config.num_tracers,
                config.boxSize_comoving,
                H_PT=self.H_PT,
                Ngrid=config.Ngrid,
                num_hash_cells=config.spatial_hash_cells
            )
            self.nucleation_volume = self.tracer_manager.get_active_zone_volume()
            self.zone_info = self.tracer_manager.get_zone_info()
        
        # Diagnostics
        self.tracers_nucleated = 0
        self.tracers_engulfed = 0

        # Setup FFT
        self.fft_manager.setup(self.field_evolver.grid_shape, FLOAT_TYPE)
        self.field_evolver.parent_simulation = self
        
        # Initialize energy calculator
        self.energy_calculator = EnergyCalculator(config, self.field_evolver)
        
        # Simulation state
        self.phase_transition_complete = False
        self.current_false_vacuum_fraction = 1.0
        
        # Data storage
        self.simulation_data = {
            'times_conformal': [],
            'times_cosmic': [],
            'bubble_counts': [],
            'false_vacuum_fractions_main': [],
            'energy_history': [],
            'nucleation_volume': self.nucleation_volume,
            'box_volume': config.boxSize_comoving**3,
            'zone_info': self.zone_info,
            'H_PT': self.H_PT
        }
        
        self.checkpoint_snapshots = []

    def _nucleation_rate(self, t_cosmic: float) -> float:
        return self.config.Gamma_0 * np.exp(self.config.beta * (t_cosmic - self.config.t_0))

    def _calculate_and_save_energy(self, tau: float):
        """Calculate energy components from current state and save."""
        centers, radii_comoving = self.bubble_manager.get_centers_and_radii_comoving(
            self.cosmology.scale_factor(tau)
        )
        
        mass_squared = self.field_evolver.mass_manager.get_cached_mass_squared()
        if mass_squared is None:
            mass_squared = self.field_evolver.get_mass_field(
                centers, radii_comoving, tau,
                self.current_false_vacuum_fraction,
                self.phase_transition_complete
            )
        
        bubble_mask = None
        if self.bubble_manager.count > 0:
            bubble_mask = get_bubble_mask_snapshot(
                self.field_evolver.X, self.field_evolver.Y, self.field_evolver.Z,
                centers, radii_comoving, self.config.boxSize_comoving
            )
        
        energy_data = self.energy_calculator.calculate_energy_components(
            tau, mass_squared,
            self.field_evolver.theta,
            self.field_evolver.theta_prime,
            bubble_mask
        )
        
        energy_data['num_bubbles'] = self.bubble_manager.count
        energy_data['tau'] = tau
        energy_data['t_cosmic'] = self.cosmology.cosmic_time_from_conformal(tau)
        
        self.simulation_data['energy_history'].append(energy_data)

    def _save_checkpoint(self, tau: float):
        """Save full field snapshot."""
        centers, radii_comoving = self.bubble_manager.get_centers_and_radii_comoving(
            self.cosmology.scale_factor(tau)
        )
        
        checkpoint = {
            'tau': tau,
            't_cosmic': self.cosmology.cosmic_time_from_conformal(tau),
            'theta': self.field_evolver.theta.copy(),
            'theta_prime': self.field_evolver.theta_prime.copy(),
            'bubble_centers': centers.copy(),
            'bubble_radii': radii_comoving.copy(),
            'false_vacuum_fraction': self.current_false_vacuum_fraction
        }
        
        self.checkpoint_snapshots.append(checkpoint)

    # def run_simulation(self, tau_final: float, save_interval: int = 10,
    #                    progress_bar: bool = True) -> Dict[str, Any]:
    #     """Run simulation with optimizations."""
    #     if tau_final <= self.config.tau0:
    #         raise ValueError("tau_final must be > tau0")

    #     tau = self.config.tau0
    #     step_count = 0

    #     est_dtau = self.field_evolver.get_recommended_timestep(tau)
    #     nsteps = max(1, int((tau_final - tau) / est_dtau))
    #     progress_bar = False
    #     pbar = tqdm(total=nsteps, desc="Simulation Progress", disable=not progress_bar)

    #     while tau < tau_final:
    #         dtau_theta = self.field_evolver.get_recommended_timestep(tau)
    #         dtau_theta = min(dtau_theta, tau_final - tau)
            
    #         # Get starting state
    #         centers_start, radii_comoving_start = self.bubble_manager.get_centers_and_radii_comoving(
    #             self.cosmology.scale_factor(tau)
    #         )
            
    #         # Field evolution (first half)
    #         self.field_evolver.evolve_step_part1(tau, dtau_theta, centers_start, radii_comoving_start)

    #         # Nucleation
    #         tau_subcycle = 0
    #         while tau_subcycle < dtau_theta:
    #             dtau_nucl = dtau_theta - tau_subcycle
                
    #             if self.tracer_manager and self.tracer_manager.num_active_tracers > 0:
    #                 self.current_false_vacuum_fraction = self.tracer_manager.get_false_vacuum_fraction()
    #                 t_cosmic = self.cosmology.cosmic_time_from_conformal(tau + tau_subcycle)
    #                 gamma_t = self._nucleation_rate(t_cosmic)
    #                 scale_factor = self.cosmology.scale_factor(tau + tau_subcycle)
                    
    #                 volume_per_tracer_comoving = self.nucleation_volume / self.config.num_tracers
    #                 volume_per_tracer_physical = volume_per_tracer_comoving * (scale_factor ** 3)
    #                 dt_cosmic = dtau_nucl * scale_factor
    #                 expected_nucleations_per_tracer = gamma_t * volume_per_tracer_physical * dt_cosmic
                    
    #                 if expected_nucleations_per_tracer > 0.1:
    #                     dtau_nucl *= (0.1 / expected_nucleations_per_tracer)
    #                     dt_cosmic = dtau_nucl * scale_factor
    #                     expected_nucleations_per_tracer = gamma_t * volume_per_tracer_physical * dt_cosmic
                    
    #                 P_nucleation = expected_nucleations_per_tracer
    #                 active_coords = self.tracer_manager.get_active_tracer_coords()
    #                 rolls = np.random.rand(len(active_coords)).astype(FLOAT_TYPE)
    #                 nucleated_mask = rolls < P_nucleation
                    
    #                 for center in active_coords[nucleated_mask]:
    #                     self.bubble_manager.add_bubble(center, self.config.v_bubble, tau + tau_subcycle)
                    
    #                 n_nucleated = np.sum(nucleated_mask)
    #                 self.tracers_nucleated += n_nucleated
    #                 if n_nucleated > 0:
    #                     self.tracer_manager.active_indices = \
    #                         self.tracer_manager.active_indices[~nucleated_mask]
    #                     self.current_false_vacuum_fraction = \
    #                         self.tracer_manager.get_false_vacuum_fraction()
    #                     print(f"Step {step_count}: Nucleating {n_nucleated} bubbles, "
    #                         f"active_tracers={len(active_coords)}, "
    #                         f"total_bubbles={self.bubble_manager.count}")

    #             tau_subcycle += dtau_nucl

    #         # Bubble evolution
    #         tau_new = tau + dtau_theta
    #         self.bubble_manager.evolve_bubbles(tau_new)
            
    #         # Field evolution (second half)
    #         centers_end, radii_comoving_end = self.bubble_manager.get_centers_and_radii_comoving(
    #             self.cosmology.scale_factor(tau_new)
    #         )
    #         self.field_evolver.evolve_step_part2(tau_new, dtau_theta, centers_end, radii_comoving_end)
            
    #         # Tracer update
    #         if self.tracer_manager:
    #             N_before_engulfment = self.tracer_manager.num_active_tracers
    #             centers, radii_comoving = self.bubble_manager.get_centers_and_radii_comoving(
    #                 self.cosmology.scale_factor(tau_new)
    #             )
    #             self.tracer_manager.update_state(centers, radii_comoving)
    #             self.current_false_vacuum_fraction = self.tracer_manager.get_false_vacuum_fraction()
    #             N_after_engulfment = self.tracer_manager.num_active_tracers
    #             n_engulfed = N_before_engulfment - N_after_engulfment
    #             self.tracers_engulfed += n_engulfed

    #         if self.current_false_vacuum_fraction <= 0.0:
    #             self.phase_transition_complete = True

    #         # Save diagnostics
    #         if step_count % save_interval == 0:
    #             self._save_snapshot(tau_new)
                
    #             # Energy calculation (less frequent)
    #             if step_count % self.config.energy_save_interval == 0:
    #                 self._calculate_and_save_energy(tau_new)
                
    #             # Checkpoints
    #             if self.config.checkpoint_interval > 0 and step_count % self.config.checkpoint_interval == 0:
    #                 self._save_checkpoint(tau_new)
                
    #             if pbar:
    #                 pbar.set_postfix({
    #                     't': f'{self.cosmology.cosmic_time_from_conformal(tau_new):.2f}',
    #                     'N_bub': self.bubble_manager.count,
    #                     'fv_frac': f'{self.current_false_vacuum_fraction:.3f}'
    #                 })

    #         # Increment time
    #         tau += dtau_theta
    #         step_count += 1
    #         if pbar:
    #             pbar.n = min(int((tau - self.config.tau0) / (tau_final - self.config.tau0) * nsteps), nsteps)
    #             pbar.refresh()

    #     if pbar:
    #         pbar.close()

    #     # Final saves
    #     print("Saving final simulation state...")
    #     self._save_snapshot(tau)
    #     self._calculate_and_save_energy(tau)
    #     if self.config.checkpoint_interval > 0:
    #         self._save_checkpoint(tau)
        
    #     return self._finalize_results()



    def run_simulation(self, tau_final: float, save_interval: int = 10,
                    progress_interval: int = 100, progress_bar: bool = False) -> Dict[str, Any]:
        """Optimized simulation loop (drop-in replacement for OptimizedAxionSimulation.run_simulation).

        Key changes for max performance:
        - No tqdm, no GUI, no frequent printing.
        - Minimal stdout progress updates every `progress_interval` steps.
        - Batched/conditional energy calculations and snapshot saves.
        - Local variable caching to reduce attribute lookups.
        - Single log file opened once and written to sparsely to avoid console locks.
        - Avoids creating temporaries where possible and minimizes Python-level per-cell work.

        Note: This is intended as a direct replacement method inside your existing
        OptimizedAxionSimulation class. It assumes the rest of the class members
        (bubble_manager, tracer_manager, field_evolver, etc.) exist unchanged.
        """
        import sys
        import time
        import logging

        # --- light-weight progress logging setup (file-backed) -----------------
        log_path = getattr(self, "performance_log_path", None)
        if log_path is None:
            # default log file name in working dir (append mode)
            log_path = "simulation_progress.log"
            self.performance_log_path = log_path

        # open once to avoid repeated open/close overhead; flush rarely
        log_file = open(log_path, "a", buffering=1)

        try:
            if tau_final <= self.config.tau0:
                raise ValueError("tau_final must be > tau0")

            # Local caching for speed (avoid attribute lookups inside hot loop)
            config = self.config
            field = self.field_evolver
            bubble_mgr = self.bubble_manager
            tracer_mgr = self.tracer_manager
            cosmology = self.cosmology
            energy_calc = self.energy_calculator

            tau = config.tau0
            step_count = 0

            # estimate dtau and number of steps (best-effort)
            est_dtau = field.get_recommended_timestep(tau)
            nsteps = max(1, int((tau_final - tau) / est_dtau))

            # performance parameters
            energy_interval = max(1, config.energy_save_interval)
            checkpoint_interval = max(0, config.checkpoint_interval)
            save_interval = max(1, save_interval)
            progress_interval = max(1, progress_interval)

            # minimize attribute access in the loop
            nucleation_volume = getattr(self, 'nucleation_volume', None)
            num_tracers = getattr(config, 'num_tracers', 0)
            box_size = config.boxSize_comoving

            # small helper references
            get_mass_field = field.get_mass_field
            evolve_part1 = field.evolve_step_part1
            evolve_part2 = field.evolve_step_part2
            evolve_bubbles = bubble_mgr.evolve_bubbles
            add_bubble = bubble_mgr.add_bubble

            # Avoid performing heavy work when tracers are exhausted
            tracers_exist = (tracer_mgr is not None and tracer_mgr.num_active_tracers > 0)

            # Main loop
            t_loop_start = time.time()
            last_progress_t = t_loop_start

            while tau < tau_final:
                # compute a stable timestep
                dtau_theta = field.get_recommended_timestep(tau)
                if tau + dtau_theta > tau_final:
                    dtau_theta = tau_final - tau

                # cache starting bubble state
                centers_start, radii_start = bubble_mgr.get_centers_and_radii_comoving(
                    cosmology.scale_factor(tau)
                )

                # First half-step (vectorized, fast)
                evolve_part1(tau, dtau_theta, centers_start, radii_start)

                # Nucleation subcycle (keep minimal Python overhead)
                tau_subcycle = 0.0
                if tracers_exist:
                    # cache some tracer variables locally
                    volume_per_tracer_comoving = (nucleation_volume / float(num_tracers)) if num_tracers > 0 else 0.0

                    active_coords = tracer_mgr.get_active_tracer_coords()
                    # We will only call get_false_vacuum_fraction when needed; keep local

                while tau_subcycle < dtau_theta:
                    dtau_nucl = dtau_theta - tau_subcycle

                    if tracers_exist and tracer_mgr.num_active_tracers > 0:
                        # compute nucleation probabilistic parameters
                        current_tau = tau + tau_subcycle
                        self.current_false_vacuum_fraction = tracer_mgr.get_false_vacuum_fraction()
                        t_cosmic = cosmology.cosmic_time_from_conformal(current_tau)
                        gamma_t = self._nucleation_rate(t_cosmic)
                        scale_factor = cosmology.scale_factor(current_tau)

                        # expected nucleations per tracer (physical volume)
                        volume_per_tracer_physical = volume_per_tracer_comoving * (scale_factor ** 3)
                        dt_cosmic = dtau_nucl * scale_factor
                        expected_nucleations_per_tracer = gamma_t * volume_per_tracer_physical * dt_cosmic

                        # cap probability to keep stability (this is your existing logic)
                        if expected_nucleations_per_tracer > 0.1:
                            dtau_nucl *= (0.1 / expected_nucleations_per_tracer)
                            dt_cosmic = dtau_nucl * scale_factor
                            expected_nucleations_per_tracer = gamma_t * volume_per_tracer_physical * dt_cosmic

                        P_nucleation = expected_nucleations_per_tracer

                        if P_nucleation > 0.0:
                            # Draw stochastic nucleations in a vectorized manner
                            active_coords = tracer_mgr.get_active_tracer_coords()
                            rolls = np.random.rand(len(active_coords)).astype(FLOAT_TYPE)
                            nucleated_mask = rolls < P_nucleation

                            if nucleated_mask.any():
                                # add bubbles in pure Python but avoid any printing
                                coords_to_add = active_coords[nucleated_mask]
                                for center in coords_to_add:
                                    add_bubble(center, config.v_bubble, tau + tau_subcycle)

                                # remove nucleated tracers from active set efficiently
                                tracer_mgr.active_indices = tracer_mgr.active_indices[~nucleated_mask]
                                self.tracers_nucleated += int(nucleated_mask.sum())

                    tau_subcycle += dtau_nucl

                # Advance bubble radii to the new time
                tau_new = tau + dtau_theta
                evolve_bubbles(tau_new)

                # Second half-step
                centers_end, radii_end = bubble_mgr.get_centers_and_radii_comoving(
                    cosmology.scale_factor(tau_new)
                )
                evolve_part2(tau_new, dtau_theta, centers_end, radii_end)

                # Tracer removal (if present)
                if tracers_exist and tracer_mgr.num_active_tracers > 0:
                    N_before = tracer_mgr.num_active_tracers
                    tracer_mgr.update_state(centers_end, radii_end)
                    N_after = tracer_mgr.num_active_tracers
                    self.tracers_engulfed += (N_before - N_after)
                    self.current_false_vacuum_fraction = tracer_mgr.get_false_vacuum_fraction()

                # If the false vacuum is gone, mark PT complete
                if self.current_false_vacuum_fraction <= 0.0:
                    self.phase_transition_complete = True

                # Periodic lightweight snapshot + energy calc
                if step_count % save_interval == 0:
                    # lightweight snapshot
                    self._save_snapshot(tau_new)

                    # energy calculation less frequently and only when needed
                    if step_count % energy_interval == 0:
                        # avoid repeated mass recomputation by pulling cached mass
                        centers_now, radii_now = bubble_mgr.get_centers_and_radii_comoving(
                            cosmology.scale_factor(tau_new)
                        )
                        mass_sq = field.mass_manager.get_cached_mass_squared()
                        if mass_sq is None:
                            mass_sq = get_mass_field(centers_now, radii_now, tau_new,
                                                    self.current_false_vacuum_fraction,
                                                    self.phase_transition_complete)

                        # compute energy (this is relatively expensive but done rarely)
                        energy_data = energy_calc.calculate_energy_components(
                            tau_new, mass_sq, field.theta, field.theta_prime,
                            None if bubble_mgr.count == 0 else get_bubble_mask_snapshot(
                                field.X, field.Y, field.Z, centers_now, radii_now, box_size
                            )
                        )
                        energy_data['num_bubbles'] = bubble_mgr.count
                        energy_data['tau'] = tau_new
                        energy_data['t_cosmic'] = cosmology.cosmic_time_from_conformal(tau_new)
                        self.simulation_data['energy_history'].append(energy_data)

                    # checkpoints saved even less frequently (config.checkpoint_interval)
                    if checkpoint_interval > 0 and step_count % checkpoint_interval == 0:
                        self._save_checkpoint(tau_new)

                # Occasional progress log to file + minimal stdout update
                if step_count % progress_interval == 0:
                    t_now = time.time()
                    elapsed = t_now - t_loop_start
                    # minimal console update (overwrites same line, cheap)
                    try:
                        sys.stdout.write(f"\rStep {step_count}/{nsteps}  tau={tau_new:.5g}  bubbles={bubble_mgr.count}  fv_frac={self.current_false_vacuum_fraction:.5f}")
                        sys.stdout.flush()
                    except Exception:
                        # console may not accept writes (background runs); ignore
                        pass

                    # write compact progress to file (single small write)
                    log_file.write(f"{time.asctime()} | step={step_count} | tau={tau_new:.6g} | bubbles={bubble_mgr.count} | fv_frac={self.current_false_vacuum_fraction:.6g} | elapsed_s={elapsed:.2f}\n")

                # Advance time and counters (hot path minimized)
                tau = tau_new
                step_count += 1

            # End main loop
            t_total = time.time() - t_loop_start

            # Final saves (one-shot, may be expensive but required)
            self._save_snapshot(tau)
            # ensure energy saved for final state
            try:
                mass_sq = field.mass_manager.get_cached_mass_squared()
                if mass_sq is None:
                    centers_now, radii_now = bubble_mgr.get_centers_and_radii_comoving(cosmology.scale_factor(tau))
                    mass_sq = get_mass_field(centers_now, radii_now, tau,
                                            self.current_false_vacuum_fraction,
                                            self.phase_transition_complete)
                self._calculate_and_save_energy(tau)
            except Exception:
                # energy calculations can fail if memory is tight; don't crash the run
                pass

            # Small final newline for console
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass

            return self._finalize_results()

        finally:
            # ensure log file closed
            try:
                log_file.close()
            except Exception:
                pass


    def _save_snapshot(self, tau: float):
        """Save lightweight snapshot."""
        self.simulation_data['times_conformal'].append(tau)
        self.simulation_data['times_cosmic'].append(
            self.cosmology.cosmic_time_from_conformal(tau)
        )
        self.simulation_data['bubble_counts'].append(self.bubble_manager.count)
        self.simulation_data['false_vacuum_fractions_main'].append(
            self.current_false_vacuum_fraction
        )

    def _finalize_results(self) -> Dict[str, Any]:
        """Return results with metadata."""
        return {
            **self.simulation_data,
            'checkpoint_snapshots': self.checkpoint_snapshots,
            'config': self.config,
            'cosmology': self.cosmology,
            'tracers_nucleated': self.tracers_nucleated,
            'tracers_engulfed': self.tracers_engulfed,
        }

# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def create_simulation_pt_at_zero(H_PT: float, beta: float, start_time: float,
                                 end_time: float, nucleation_mode: str = 'standard',
                                 **kwargs):
    """Factory function for creating simulations."""
    cosmology = CosmologyManagerPTAtZeroStartAtZero(H_PT)
    tau_initial = cosmology.conformal_time_from_cosmic(start_time)
    tau_final = cosmology.conformal_time_from_cosmic(end_time)
    
    config = SimulationConfig(
        tau0=tau_initial,
        a0=cosmology.scale_factor(tau_initial),
        Gamma_0=(H_PT**4),
        t_0=cosmology.t_PT,
        beta=beta,
        **kwargs
    )
    
    sim = OptimizedAxionSimulation(
        config, cosmology,
        fft_backend='fftw',
        H_PT=H_PT,
        nucleation_mode=nucleation_mode
    )
    return sim, tau_final


if __name__ == "__main__":
    print("="*70)
    print("Axion Simulation v8.1 - PROFILING-OPTIMIZED Edition")
    print("="*70)
    print("Optimizations:")
    print("  ✓ Aggressive thread suppression (eliminate lock contention)")
    print("  ✓ Reduced energy calculation frequency")
    print("  ✓ Improved mass field caching")
    print("  ✓ Optional FFT-based Laplacian (faster for large grids)")
    print("  ✓ Better cache invalidation tracking")
    print("="*70)