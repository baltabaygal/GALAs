# CMB Constraints

This folder is for the **pre-inflation, non-equilibrium** axion case only.

Scope in this folder:

- one homogeneous initial angle `\theta_0`
- inflationary fluctuation
  \[
  \delta\theta = \frac{H_I}{2\pi f_\phi}
  \]
- standard pre-inflation isocurvature amplitude
- PT modification through the derivative
  \[
  \partial_{\theta_0}\ln \xi
  \]

Not included here yet:

- equilibrium / stochastic-inflation initial distribution
- averaging over `P_{\rm eq}`
- full CMB likelihood machinery

Files:

- [calculation.md](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/cmbconstraints/calculation.md)
  - equations and conventions
- [calc_preinflation_noneq.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/cmbconstraints/calc_preinflation_noneq.py)
  - standalone calculator for `P_S(k_*)` in the non-equilibrium pre-inflation case
- [scan_preinflation_noneq_bound.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/cmbconstraints/scan_preinflation_noneq_bound.py)
  - scans a `\theta_0` grid and converts the response into a bound on `H_I/f_\phi`
- [scan_nopt_reference_bound.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/cmbconstraints/scan_nopt_reference_bound.py)
  - scans the standard noPT bound and compares the exact anharmonic result to the harmonic reference
