"""
Axion Parameter Sweep Utility v1.0
=================================

Performs a serial parameter sweep over H_PT, beta/H, and initial theta0 values.
Utilizes the Axion Simulation Core v1.0 for high-performance field evolution.

Features:
---------
- Automatic result saving in JSON format.
- Progress tracking with summary generation.
- Error handling and logging for individual realizations.
"""

import os
import json
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, List, Tuple
from axion_sim_v1p0 import create_simulation, FLOAT_TYPE

# ==============================================================================
# UTILITIES
# ==============================================================================

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

# ==============================================================================
# WORKER LOGIC
# ==============================================================================

def run_single_simulation(params: Tuple) -> Dict[str, Any]:
    """
    Executes a single simulation instance based on provided parameters.
    """
    H_PT, beta_over_H, theta0, with_pt, realization = params
    
    try:
        # Determine physics mode
        if with_pt:
            beta_h = beta_over_H
            enable_nucleation = True
            mb_value = 0.0
        else:
            beta_h = 0.0
            enable_nucleation = False
            mb_value = 1.0  # No mass shift

        # Initialize simulation via factory
        sim = create_simulation(
            H_PT=H_PT,
            beta_H=beta_h,
            Ngrid=32,
            num_tracers=100000 if enable_nucleation else 1000,
            enable_nucleation=enable_nucleation,
            mb=mb_value,
            theta0_initial=theta0,
            v_bubble=0.6,
            energy_save_interval=10
        )

        # Run until final time (standard 10 Hubble times)
        tau_final = sim.cosmo.conformal_time(10.0 / H_PT)
        history = sim.run(tau_final)

        if history['energy']:
            final_e = history['energy'][-1]
            config_dict = asdict(sim.config)

            return convert_to_native({
                'status': 'SUCCESS',
                'params': {
                    'H_PT': H_PT,
                    'beta_over_H': beta_over_H,
                    'theta0': theta0,
                    'with_pt': with_pt,
                    'realization': realization
                },
                'results': {
                    'final_time_cosmic': history['t'][-1],
                    'final_fv_fraction': history['fv_frac'][-1],
                    'kinetic': final_e['kin'],
                    'gradient': final_e['grad'],
                    'potential': final_e['pot'],
                    'total': final_e['kin'] + final_e['grad'] + final_e['pot'],
                    'num_bubbles': sim.bubbles.count
                },
                'config': config_dict
            })

        return {'status': 'FAILED', 'error': 'No energy history recorded'}

    except Exception as e:
        import traceback
        return {
            'status': 'ERROR',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# ==============================================================================
# SWEEP MANAGER
# ==============================================================================

class ParameterSweep:
    def __init__(self, output_dir: str = 'sweep_results_v1p0'):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def run(self, H_PT_list: List[float], beta_H_list: List[float], 
            theta0_list: List[float], realizations: int):
        
        tasks = []
        # 1. No-PT Reference Tasks
        for h in H_PT_list:
            for t in theta0_list:
                tasks.append((h, 0.0, t, False, 0))
        
        # 2. Nucleation Tasks
        for h in H_PT_list:
            for b in beta_H_list:
                for t in theta0_list:
                    for r in range(realizations):
                        tasks.append((h, b, t, True, r))

        print(f"Starting sweep: {len(tasks)} tasks queued.")
        
        summary = {'successful': 0, 'failed': 0, 'results_files': []}

        for i, task in enumerate(tasks, 1):
            h, b, t, pt, r = task
            mode = "PT" if pt else "Ref"
            print(f"[{i}/{len(tasks)}] Mode: {mode} | H_PT: {h:.2f} | beta/H: {b:.1f} | theta0: {t:.2f} | Real: {r}")
            
            result = run_single_simulation(task)
            
            if result['status'] == 'SUCCESS':
                summary['successful'] += 1
                fname = f"{mode}_H{h:.3f}_B{b:.1f}_T{t:.3f}_R{r}.json"
                fpath = os.path.join(self.data_dir, fname)
                with open(fpath, 'w') as f:
                    json.dump(result, f, indent=2)
                summary['results_files'].append(fname)
            else:
                summary['failed'] += 1
                print(f"  !! Error: {result.get('error')}")

        # Save Summary
        with open(os.path.join(self.output_dir, 'sweep_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nSweep Complete.")
        print(f"Results saved to: {self.output_dir}")
        print(f"Success: {summary['successful']} | Failed: {summary['failed']}")

# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Example Sweep Parameters (Downscaled for test)
    H_PT_VALS = [0.07,0.7,1.3,1.9]
    THETA0_VALS = [
        np.pi/11,
        np.pi/7 + np.pi/12,
        np.pi/3 + np.pi/13,
        np.pi/2.2 + np.pi/12.9,
        2*np.pi/3.5 + np.pi/11,
        5*np.pi/6 + np.pi/10
    ]
    BETA_H_VALS = [4, 8, 10, 16, 20, 32, 40]
    REALS = 3

    sweep = ParameterSweep(output_dir='production_sweep_f_test')
    sweep.run(H_PT_VALS, BETA_H_VALS, THETA0_VALS, REALS)
