"""
Sweep Analysis & Plotting Utility v1.0
=====================================

Analyzes the output of run_sweep_v1p0.py. Aggregates JSON results
and produces publication-quality summary plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class SweepAnalyzer:
    def __init__(self, sweep_dir: str):
        self.sweep_dir = sweep_dir
        self.data_dir = os.path.join(sweep_dir, 'data')
        self.results = []
        
        # Load results
        self._load_results()

    def _load_results(self):
        if not os.path.exists(self.data_dir):
            print(f"Error: {self.data_dir} not found.")
            return
            
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        print(f"Loading {len(files)} result files...")
        
        for fname in files:
            with open(os.path.join(self.data_dir, fname), 'r') as f:
                self.results.append(json.load(f))

    def plot_energy_summary(self, h_pt_target: float = None):
        """
        Plots final total energy vs beta/H for a given H_PT.
        """
        # Group by with_pt
        ref_data = [r for r in self.results if not r['params']['with_pt']]
        pt_data = [r for r in self.results if r['params']['with_pt']]
        
        if h_pt_target is not None:
            pt_data = [r for r in pt_data if abs(r['params']['H_PT'] - h_pt_target) < 1e-5]
            ref_data = [r for r in ref_data if abs(r['params']['H_PT'] - h_pt_target) < 1e-5]
            
        if not pt_data:
            print("No PT data found for selected criteria.")
            return

        # Prepare plots
        plt.figure(figsize=(10, 6))
        
        # Extract beta/H and total energy
        betas = np.array([r['params']['beta_over_H'] for r in pt_data])
        energies = np.array([r['results']['total'] for r in pt_data])
        thetas = np.array([r['params']['theta0'] for r in pt_data])
        
        # Unique thetas for legend
        unique_thetas = np.unique(thetas)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_thetas)))
        
        for i, theta in enumerate(unique_thetas):
            mask = (thetas == theta)
            b_vals = betas[mask]
            e_vals = energies[mask]
            
            # Sort by beta for clean line
            sort_idx = np.argsort(b_vals)
            plt.plot(b_vals[sort_idx], e_vals[sort_idx], 'o-', label=f'$\\theta_0 = {theta:.2f}$', color=colors[i])
            
            # Reference baseline if available
            ref_theta = [r for r in ref_data if abs(r['params']['theta0'] - theta) < 1e-5]
            if ref_theta:
                baseline = ref_theta[0]['results']['total']
                plt.axhline(baseline, color=colors[i], linestyle='--', alpha=0.5)

        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('$\\beta / H$')
        plt.ylabel('Final Total Energy Density')
        plt.title(f'Phase Transition Impact on Final Energy (H_PT={h_pt_target})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_name = f"summary_HPT_{h_pt_target}.png" if h_pt_target else "summary_all.png"
        plt.savefig(save_name, dpi=300)
        print(f"Summary plot saved to: {save_name}")

if __name__ == "__main__":
    # Point this to your sweep results directory
    SWEEP_DIR = "production_sweep_v1p0"
    if os.path.exists(SWEEP_DIR):
        analyzer = SweepAnalyzer(SWEEP_DIR)
        # Try to plot for one of the H_PT values we used
        analyzer.plot_energy_summary(h_pt_target=0.07)
    else:
        print(f"Sweep directory {SWEEP_DIR} not found. Run the sweep first!")
