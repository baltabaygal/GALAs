"""
Single Axion Simulation Runner & Visualizer v1.0
==============================================

Runs a single instance of the axion simulation, saves the full history,
and produces an immediate visualization of the energy evolution.
Ideal for quick tests and parameter debugging.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from axion_sim_v1p0 import create_simulation

def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

def run_and_plot(H_PT=1.0, beta_H=10.0, theta0=2.0, Ngrid=32, tau_max=5.0):
    print(f"--- Running Single Simulation ---")
    print(f"Parameters: H_PT={H_PT}, beta/H={beta_H}, theta0={theta0}, N={Ngrid}")
    
    # 1. Initialize and Run
    sim = create_simulation(
        H_PT=H_PT, 
        beta_H=beta_H, 
        Ngrid=Ngrid, 
        num_tracers=50000,
        enable_nucleation=True,
        energy_save_interval=5
    )
    
    tau_final = sim.cosmo.conformal_time(tau_max / H_PT)
    history = sim.run(tau_final)
    
    # 2. Extract Data
    times = np.array(history['t'])
    # Scale factor for radiation dominance: a(t) = sqrt(2 * H_PT * t)
    # But sim uses a(tau) = H_PT * tau. Let's use history to be consistent.
    
    energies = history['energy']
    kin = np.array([e['kin'] for e in energies])
    pot = np.array([e['pot'] for e in energies])
    grad = np.array([e['grad'] for e in energies])
    total = kin + pot + grad
    
    fv_frac = np.array(history['fv_frac'])
    
    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Energy Plot
    ax1.plot(times, kin, label='Kinetic', alpha=0.8)
    ax1.plot(times, pot, label='Potential', alpha=0.8)
    ax1.plot(times, grad, label='Gradient', alpha=0.8)
    ax1.plot(times, total, 'k--', label='Total Energy', linewidth=2)
    ax1.set_ylabel('Energy Density')
    ax1.set_yscale('log')
    ax1.set_title(f'Axion Energy Evolution (N={Ngrid}, H_PT={H_PT}, $\\beta/H$={beta_H})')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # False Vacuum Plot
    ax2.plot(times, fv_frac, color='red', linewidth=2, label='False Vacuum Fraction')
    ax2.set_ylabel('FV Fraction')
    ax2.set_xlabel('Cosmic Time (t)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "single_sim_test.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    
    # 4. Save History
    # (Numpy arrays handled by converting to lists)
    history_json = {
        'params': {'H_PT': H_PT, 'beta_H': beta_H, 'theta0': theta0, 'Ngrid': Ngrid},
        't': times.tolist(),
        'fv_frac': fv_frac.tolist(),
        'energy': energies # Already a list of dicts
    }
    
    with open("single_sim_history.json", "w") as f:
        json.dump(convert_to_native(history_json), f, indent=2)
    print("History saved to: single_sim_history.json")

if __name__ == "__main__":
    # Standard testing parameters
    run_and_plot(H_PT=0.7, beta_H=15.0, theta0=np.pi/3, Ngrid=32, tau_max=2.0)
