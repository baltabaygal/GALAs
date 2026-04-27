"""
Serial parameter sweep over H_PT, beta/H, theta0_initial, and multiple realizations.
Runs one simulation at a time, saves result, then starts next.

UPDATED FOR V8 - Compatible with axion_sim_v8.py (optimized version)
Key changes:
- Energy is calculated on-the-fly, no post-processing needed
- Uses energy_history from results directly
- Adapted for lightweight data storage
"""

import numpy as np
import matplotlib.pyplot as plt
from axion_sim_v9 import (
    create_simulation_pt_at_zero,
    calculate_mass_field_snapshot_subsampled,
    EnergyCalculator,
    get_bubble_mask_snapshot
)
import os
import json
from dataclasses import asdict


def run_single_sim_worker(params):
    """
    Worker function for a single simulation.
    """
    H_PT, beta_over_H, theta0, with_pt, realization = params

    try:
        if with_pt:
            beta = beta_over_H * H_PT
            enable_nucleation = True
            mb_value = 0.0
        else:
            beta = 0.0
            enable_nucleation = False
            mb_value = 1.0

        sim, tau_final = create_simulation_pt_at_zero(
            H_PT=H_PT,
            beta=beta,
            start_time=0.0,
            end_time=10.0 / H_PT,
            nucleation_mode='standard',
            Ngrid=64,
            num_tracers=100000 if enable_nucleation else 1000,
            enable_nucleation=enable_nucleation,
            mb=mb_value,
            theta0_initial=theta0,
            v_bubble=0.6,
            spatial_hash_cells=8,
            energy_save_interval=10,
            checkpoint_interval=0
        )

        results = sim.run_simulation(tau_final, progress_bar=False, save_interval=10)

        # V8: Energy is already calculated and stored in energy_history
        if results['times_cosmic'] and results['energy_history']:
            final_time = results['config'].m0 * results['times_cosmic'][-1]
            
            # Get final energy from the stored history
            final_energy = results['energy_history'][-1]
            
            config_dict = asdict(sim.config)

            return convert_to_native_types({
                'H_PT': H_PT,
                'beta_over_H': beta_over_H,
                'theta0_initial': theta0,
                'with_pt': with_pt,
                'realization': realization,
                'final_time': final_time,
                'kinetic': final_energy['kinetic_energy'],
                'gradient': final_energy['gradient_energy'],
                'potential': final_energy['potential_energy'],
                'total': final_energy['total_energy'],
                'scale_factor': final_energy['scale_factor'],
                'num_bubbles': sim.bubble_manager.count,
                'status': 'SUCCESS',
                'config': config_dict
            })

        return {
            'status': 'FAILED', 
            'H_PT': H_PT, 
            'beta_over_H': beta_over_H,
            'theta0_initial': theta0, 
            'with_pt': with_pt, 
            'realization': realization
        }

    except Exception as e:
        import traceback
        return {
            'status': 'ERROR',
            'H_PT': H_PT,
            'beta_over_H': beta_over_H,
            'theta0_initial': theta0,
            'with_pt': with_pt,
            'realization': realization,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


class SerialParameterSweep:
    """Serial parameter sweep manager with theta sweep."""

    def __init__(self, output_dir='serial_theta_sweep_results_v8'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'representative_plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'final_energies'), exist_ok=True)

    def run_serial_sweep(self, H_PT_values, beta_over_H_values, theta0_values, num_realizations):
        """
        Run parameter sweep serially.
        """
        print("="*70)
        print("SERIAL PARAMETER SWEEP WITH THETA (V8 OPTIMIZED)")
        print("="*70)
        print(f"H_PT values: {len(H_PT_values)}")
        print(f"Beta/H values: {len(beta_over_H_values)}")
        print(f"Theta0 values: {len(theta0_values)}")
        print(f"Realizations: {num_realizations}")

        num_no_pt = len(H_PT_values) * len(theta0_values)
        num_pt = len(H_PT_values) * len(beta_over_H_values) * len(theta0_values) * num_realizations
        total_sims = num_no_pt + num_pt

        print(f"Total simulations: {total_sims}")
        print(f"  - No-PT: {num_no_pt}")
        print(f"  - With PT: {num_pt}")
        print("="*70 + "\n")

        tasks = []

        # No-PT simulations
        for H_PT in H_PT_values:
            for theta0 in theta0_values:
                tasks.append((H_PT, 0.0, theta0, False, 0))

        # With-PT simulations
        for H_PT in H_PT_values:
            for beta_over_H in beta_over_H_values:
                for theta0 in theta0_values:
                    for real in range(num_realizations):
                        tasks.append((H_PT, beta_over_H, theta0, True, real))

        print(f"Total tasks queued: {len(tasks)}\n")

        all_results = []
        failed_tasks = []

        for idx, task in enumerate(tasks, 1):
            H_PT, beta_over_H, theta0, with_pt, real = task
            print(f"\n[{idx}/{len(tasks)}] Running: H_PT={H_PT:.3f}, β/H={beta_over_H:.2f}, θ₀={theta0:.3f}, PT={with_pt}, real={real}")
            
            result = run_single_sim_worker(task)

            if result['status'] == 'SUCCESS':
                all_results.append(result)
                if with_pt:
                    filename = f'pt_HPT_{H_PT:.3f}_beta_{beta_over_H:.2f}_theta_{theta0:.3f}_real_{real}.json'
                else:
                    filename = f'no_pt_HPT_{H_PT:.3f}_theta_{theta0:.3f}.json'
                save_path = os.path.join(self.output_dir, 'final_energies', filename)
                with open(save_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  ✓ Success - Total energy: {result['total']:.6e}")
            else:
                failed_tasks.append(task)
                error_msg = result.get('error', result['status'])
                print(f"  ✗ Failed: {error_msg}")
                if 'traceback' in result:
                    print(f"  Traceback:\n{result['traceback']}")

        summary = {
            'successful': len(all_results),
            'failed': len(failed_tasks),
            'failed_tasks': [
                {'H_PT': t[0], 'beta_over_H': t[1], 'theta0': t[2], 'with_pt': t[3], 'realization': t[4]}
                for t in failed_tasks
            ],
            'sweep_parameters': {
                'H_PT_values': [float(x) for x in H_PT_values],
                'beta_over_H_values': [float(x) for x in beta_over_H_values],
                'theta0_values': [float(x) for x in theta0_values],
                'num_realizations': num_realizations
            }
        }

        with open(os.path.join(self.output_dir, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Serial sweep complete!")
        print(f"  Successful: {len(all_results)}/{len(tasks)}")
        print(f"  Failed: {len(failed_tasks)}/{len(tasks)}")
        print(f"{'='*70}\n")

        return all_results, failed_tasks

    def generate_representative_plots_sequential(self, H_PT_values, beta_over_H_values, theta0_values):
        """
        Generate representative plots sequentially after main sweep.
        V8: Energy is calculated on-the-fly, no post-processing needed.
        """
        print("\nGenerating representative plots...")

        if len(theta0_values) > 3:
            plot_thetas = [theta0_values[0], theta0_values[len(theta0_values)//2], theta0_values[-1]]
        else:
            plot_thetas = theta0_values

        for H_PT in H_PT_values:
            for theta0 in plot_thetas:
                print(f"\nPlotting H_PT={H_PT:.3f}, θ₀={theta0:.3f}")
                
                # No-PT simulation
                sim_no_pt, tau_final = create_simulation_pt_at_zero(
                    H_PT=H_PT, beta=0.0,
                    start_time=1/(2*H_PT), end_time=100.0/H_PT,
                    Ngrid=16, num_tracers=1000,
                    enable_nucleation=False, mb=1.0,
                    theta0_initial=theta0,
                    # V8-specific: Enable energy calculation
                    energy_save_interval=10,
                    checkpoint_interval=0
                )
                results_no_pt = sim_no_pt.run_simulation(tau_final, progress_bar=True, save_interval=10)
                # V8: Energy is already in energy_history, no post-processing needed!

                for beta_over_H in beta_over_H_values:
                    beta = beta_over_H * H_PT
                    print(f"  β/H={beta_over_H:.2f}")

                    # With-PT simulation
                    sim_pt, tau_final = create_simulation_pt_at_zero(
                        H_PT=H_PT, beta=beta,
                        start_time=1/(2*H_PT), end_time=100.0/H_PT,
                        Ngrid=16, num_tracers=100000,
                        enable_nucleation=True, mb=0.0,
                        theta0_initial=theta0,
                        # V8-specific: Enable energy calculation
                        energy_save_interval=10,
                        checkpoint_interval=0
                    )
                    results_pt = sim_pt.run_simulation(tau_final, progress_bar=True, save_interval=10)
                    # V8: Energy is already in energy_history!

                    self._generate_plot(results_pt, results_no_pt, H_PT, beta_over_H, theta0)

    def _generate_plot(self, results_pt, results_no_pt, H_PT, beta_over_H, theta0):
        """Generate single representative plot using V8 energy history."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # With-PT data - V8: Use energy_history directly
        config_pt = results_pt['config']
        
        # Extract data from energy_history
        energies_pt = results_pt['energy_history']
        times_pt = config_pt.m0 * np.array([e['t_cosmic'] for e in energies_pt])
        scale_factors_pt = np.array([e['scale_factor'] for e in energies_pt])
        gradient_pt = np.array([e['gradient_energy'] for e in energies_pt])
        kinetic_pt = np.array([e['kinetic_energy'] for e in energies_pt])
        potential_pt = np.array([e['potential_energy'] for e in energies_pt])

        scaled_gradient_pt = gradient_pt * (scale_factors_pt ** 3)
        scaled_matter_pt = (kinetic_pt + potential_pt) * (scale_factors_pt ** 3)

        # No-PT data - V8: Use energy_history directly
        config_no_pt = results_no_pt['config']
        
        energies_no_pt = results_no_pt['energy_history']
        times_no_pt = config_no_pt.m0 * np.array([e['t_cosmic'] for e in energies_no_pt])
        scale_factors_no_pt = np.array([e['scale_factor'] for e in energies_no_pt])
        kinetic_no_pt = np.array([e['kinetic_energy'] for e in energies_no_pt])
        potential_no_pt = np.array([e['potential_energy'] for e in energies_no_pt])

        scaled_matter_no_pt = (kinetic_no_pt + potential_no_pt) * (scale_factors_no_pt ** 3)

        # Plot
        ax.plot(times_pt, scaled_gradient_pt, 'orange', linewidth=2.5, label='Gradient (Nucleation)')
        ax.plot(times_pt, scaled_matter_pt, 'green', linewidth=2.5, label='Kinetic+Potential (Nucleation)')
        ax.plot(times_no_pt, scaled_matter_no_pt, 'purple', linewidth=2.5, linestyle='--',
                label='Kinetic+Potential (No Nucleation)')

        final_matter_pt = scaled_matter_pt[-1]
        final_matter_no_pt = scaled_matter_no_pt[-1]

        ax.axhline(final_matter_pt, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(final_matter_no_pt, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(0, color='red', linestyle=':', linewidth=2, alpha=0.8)

        ax.set_xlabel('Dimensionless Time ($m_0 t$)', fontsize=14)
        ax.set_ylabel('Scaled Energy Density ($\\rho a^3$)', fontsize=14)
        ax.set_title(f'Scaled Energy Evolution (V8 Optimized)\n$H_{{PT}} = {H_PT:.2f}$, $\\beta/H = {beta_over_H:.2f}$, $\\theta_0 = {theta0:.2f}$',
                    fontsize=16, fontweight='bold')

        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'representative_plots',
                                f'HPT_{H_PT:.3f}_beta_{beta_over_H:.2f}_theta_{theta0:.3f}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved plot: {save_path}")


if __name__ == "__main__":
    # Define sweep parameters
    H_PT_values = [0.07,0.7,1.3,1.9]
    theta0_values = [
        np.pi/11,
        np.pi/7 + np.pi/12,
        np.pi/3 + np.pi/13,
        np.pi/2.2 + np.pi/12.9,
        2*np.pi/3.5 + np.pi/11,
        5*np.pi/6 + np.pi/10
    ]
    beta_over_H_values = [4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40]
    num_realizations = 5

    sweep = SerialParameterSweep(output_dir='serial_theta_sweep_results_v1_highHstar')

    # Step 1: Run serial sweep
    all_results, failed = sweep.run_serial_sweep(
        H_PT_values, beta_over_H_values, theta0_values, num_realizations
    )

    # Step 2: Generate representative plots
    sweep.generate_representative_plots_sequential(H_PT_values, beta_over_H_values, theta0_values)

    print("\nSerial sweep with theta (V8 optimized) complete!")