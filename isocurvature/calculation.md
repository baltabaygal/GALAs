# Isocurvature Calculation Note

This note documents the **current** implementation in
[plot_isocurvature.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/plot_isocurvature.py).
It is written to make the normalization and units explicit, because these were
the main sources of mistakes in the earlier versions.

This note covers:

- how `\xi(\theta_0)` is used
- how the variance is computed
- how `t_p`, `t_{\rm onset}`, and `k_{\rm cut}` are defined
- which Planck mass is used
- which entropy-factor convention is used
- the white-noise prefactor used in `P_i(k)`
- where the `D_i(z_{\rm eq})^2` factor enters
- what the current `P(k)` plots are actually showing

## 1. Code Inputs

The script uses:

- `xi_model`:
  - [paper_codes/xi_model](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/xi_model)
- the RD percolation kernel:
  - [ode/hom_ODE/percolation.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/hom_ODE/percolation.py)
- the tabulated CDM spectrum:
  - [paper_codes/Pk_CDM.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM.dat)

The current standalone script writes into:

- [paper_codes/isocurvature/outputs](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs)

## 2. Basic Variables and Units

The script works with:

- `\theta_0` dimensionless
- `t`, `t_p`, `t_{\rm osc}`, `t_{\rm onset}` in units of `M_\phi^{-1}`
- `H_* / M_\phi` dimensionless
- `\beta / H_*` dimensionless
- `k` in either `{\rm Mpc}^{-1}` or `{\rm kpc}^{-1}` depending on the plot
- `P(k)` in either `{\rm Mpc}^3` or `{\rm kpc}^3`
- `\Delta^2(k)` dimensionless

The conversion constants in code are:

- reduced Planck mass:
  \[
  \bar M_{\rm Pl} = 2.435\times 10^{27}\ {\rm eV}
  \]
- present CMB temperature:
  \[
  T_0 = 2.35\times 10^{-4}\ {\rm eV}
  \]
- present-day entropy degrees of freedom:
  \[
  g_{s,0}=3.91
  \]
- oscillation-era relativistic degrees of freedom are now treated as
  temperature-dependent:
  \[
  g_*(T_{\rm osc}),\qquad g_{*s}(T_{\rm osc}),
  \]
  using the shared lookup table
  [gstar_lookup.json](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/common/gstar_lookup.json)
  instead of the older fixed approximation `g_* = g_{s,*} = 100`

The source stored with that lookup table is:

\[
\texttt{https://arxiv.org/pdf/1609.04979}.
\]

The eV to comoving wavenumber conversion is implemented as:

\[
1\ {\rm eV}
\;\to\;
\frac{{\rm Mpc}}{\hbar c}\ {\rm Mpc}^{-1},
\]

using:

- `HBARC_EV_M = 1.973269804e-7`
- `M_PER_MPC = 3.085677581491367e22`
- `EV_TO_MPC_INV = M_PER_MPC / HBARC_EV_M`

The Mpc/kpc conversions are:

\[
1\ {\rm Mpc}^{-1} = 10^{-3}\ {\rm kpc}^{-1},
\qquad
1\ {\rm Mpc}^3 = 10^9\ {\rm kpc}^3.
\]

In code:

- `MPC_INV_TO_KPC_INV = 1e-3`
- `MPC3_TO_KPC3 = 1e9`

## 3. Angular Domain

The current script integrates over the full physical range

\[
\theta_0 \in [0,\pi].
\]

The evaluation grid is dense near the hilltop:

- explicit points at `0, 0.05, 0.1, 0.2, 0.262`
- then log-spaced points up to `\pi`
- explicit endpoint `\pi`

The `xi_model` package is evaluated with `clip=True`, but the package was
patched earlier so that:

- `\theta_0` is accepted over the full `[0,\pi]`
- `A_0(\theta_0)` extrapolates linearly outside the fitted lattice angle range

So the current isocurvature calculation is **not** truncating the angular
integral to `[0.262, 2.88]`.

## 3.1 What `\theta_0` Dependence Is Included

The isocurvature script evaluates

\[
\xi(\theta_0;H_*,v_w,\beta/H_*)
\]

point by point on the full `\theta_0` grid through
[xi_model](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/xi_model).
So the central `\xi(\theta_0)` used in the integral includes all
`\theta_0` dependence present in the deployed compact model.

Included `\theta_0` dependence:

- `A_0(\theta_0)`:
  - yes
  - interpolated on the fitted lattice angles and linearly extrapolated outside
    that range
- `\lambda(\theta_0)`:
  - yes
  - implemented as
    \[
    \lambda(\theta_0)=A\,x(\theta_0)^\gamma,
    \qquad
    x(\theta_0)=\ln\!\left(\frac{e}{\cos^2(\theta_0/2)}\right)
    \]
- pilot `\kappa(\theta_0,v_w,H_*)`:
  - yes
  - obtained from the pilot-`\kappa` lattice table, with low-`H_*` plateau logic
    if applicable
- homogeneous baseline `f_0(\theta_0), f_\infty(\theta_0)`:
  - yes
  - through the compact `\tilde f` / `\xi_{\rm DM}` baseline used by
    `xi_model`
- potential weight `1-\cos\theta_0`:
  - yes
  - explicit in the final density

Not independently `\theta_0`-dependent in the central model:

- the global amplitude `A` and exponent `\gamma` of the `\lambda` law
- the global exponent `p`
- the global `q`
- the compact `C(v_w)` and `r(v_w)` laws:
  - these depend on `v_w`, not on `\theta_0`

So the isocurvature integral does use the full deployed central
`\theta_0` structure of `\xi_model`, but it does **not** promote every fit
parameter to an independent function of `\theta_0`.

Also note:

- the script uses the central `\xi`
- it does **not** propagate the `\xi_{\rm lo}/\xi_{\rm hi}` uncertainty band
  through the `\theta_0` integral

## 4. No-PT Anharmonic Factor

The baseline no-PT anharmonic factor is

\[
f_{\rm anh}^{\rm noPT}(\theta_0)
=
0.373\,
\left[
1-\ln\cos^2\!\left(\frac{\theta_0}{2}\right)
\right]^{1.20}.
\]

In code:

- `A0_FANH = 0.373`
- `GAMMA0_FANH = 1.20`
- `fanh_no_pt(theta)`

## 5. Density Definitions

### 5.1 No-PT

\[
\rho_{\rm noPT}(\theta_0)
=
f_{\rm anh}^{\rm noPT}(\theta_0)\,(1-\cos\theta_0).
\]

This is equivalent to setting

\[
\xi(\theta_0)=1.
\]

### 5.2 PT

For a given benchmark `(H_*/M_\phi, v_w, \beta/H_*)`,

\[
\rho_{\rm PT}(\theta_0)
=
\xi(\theta_0;H_*,v_w,\beta/H_*)\,
f_{\rm anh}^{\rm noPT}(\theta_0)\,
(1-\cos\theta_0).
\]

Important:

- the script uses the **raw** `\xi` from `xi_model`
- it does **not** renormalize by a fast-PT asymptote
- noPT is always defined by `\xi=1`

## 6. Mean Density and Variance

For any profile `\rho(\theta_0)`, the script computes

\[
\bar\rho
=
\frac{1}{\pi}\int_0^\pi \rho(\theta_0)\,d\theta_0
\]

and

\[
\langle \delta^2\rangle
=
\frac{1}{\pi}\int_0^\pi
\left(
\frac{\rho(\theta_0)-\bar\rho}{\bar\rho}
\right)^2
d\theta_0.
\]

This quantity is stored in code as:

- `var`

So `var` is the variance of the **fractional** density contrast, not the
variance of `\rho` itself.

The integration method is:

- dense precomputed `\theta_0` grid
- `np.trapezoid`

There is no `quad` in the current implementation.

## 6.1 Fast-PT Extrapolation Used in the Current `P(k)` Plot

For the current `h`-unit comparison plot, the optional
`\beta/H_* \to \infty` branch is treated as an **extrapolated**
fast-transition limit, not as a validated lattice interpolation point.

The rule used in code is:

- compute the PT variance at high but finite values
  \[
  \beta/H_* = 12,\ 20,\ 40
  \]
- fit `var` linearly as a function of `1/(\beta/H_*)`
- evaluate that fit at
  \[
  1/(\beta/H_*) = 0
  \]
  to define `var_{\infty}`

For the cutoff side, the fast-PT line assumes:

\[
t_p \to 0,
\qquad
t_{\rm onset}^{\rm PT} = t_{\rm osc},
\qquad
k_{\rm cut}^{\infty} = k_{\rm cut}^{\rm noPT}.
\]

So the extrapolated `\beta/H_*\to\infty` curve uses:

\[
P_i^{\infty}(k)

=
6\pi^2\,
\left(k_{\rm cut}^{\rm noPT}\right)^{-3}
\frac{1}{D_i(z_{\rm eq})^2}
\mathrm{var}_{\infty}\,
\Theta(k_{\rm cut}^{\rm noPT}-k).
\]

This line should therefore be read as a **controlled visualization
extrapolation**, not a directly fitted lattice-supported branch.

## 7. Percolation Time and Onset

The percolation time is

\[
t_p = t_{\rm perc}^{\rm RD}(H_*,\beta/H_*,v_w),
\]

from:

- [ode/hom_ODE/percolation.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/ode/hom_ODE/percolation.py)

The no-PT oscillation time is fixed to

\[
t_{\rm osc} = \frac{3}{2},
\]

because in radiation domination:

\[
H = \frac{1}{2t},
\qquad
3H = M_\phi
\;\Rightarrow\;
t_{\rm osc} = \frac{3}{2}\,M_\phi^{-1}.
\]

The PT onset is

\[
t_{\rm onset}^{\rm PT} = \max(t_p, t_{\rm osc}).
\]

So if `t_p < t_{\rm osc}`, the PT case does **not** start oscillating earlier
than noPT.

## 8. Cutoff Scale

### 8.1 Physical no-PT cutoff

The script uses

\[
H_{\rm osc} = \frac{M_\phi}{3},
\]

and iteratively solves

\[
T_{\rm osc}
=
\sqrt{H_{\rm osc}\,\bar M_{\rm Pl}}
\left(\frac{90}{\pi^2 g_*(T_{\rm osc})}\right)^{1/4}.
\]

Important correction:

- the code now uses the **reduced** Planck mass
  \[
  \bar M_{\rm Pl} = 2.435\times10^{27}\ {\rm eV}
  \]
  not the non-reduced `1.22e28 eV`

The present-day noPT cutoff is

\[
k_{\rm cut}^{\rm noPT}
=
H_{\rm osc}
\left(\frac{T_0}{T_{\rm osc}}\right)
\left(\frac{g_{s,0}}{g_{*s}(T_{\rm osc})}\right)^{1/3}.
\]

Substituting the implicit temperature relation,

\[
T_{\rm osc}\propto H_{\rm osc}^{1/2}\,g_*(T_{\rm osc})^{-1/4},
\]

gives the parametric dependence

\[
k_{\rm cut}^{\rm noPT}
\propto
H_{\rm osc}^{1/2}\,
g_*(T_{\rm osc})^{1/4}\,
g_{*s}(T_{\rm osc})^{-1/3}.
\]

So:

- larger `g_*(T_{\rm osc})` increases `k_{\rm cut}` through the `T_{\rm osc}`
  relation
- larger `g_{*s}(T_{\rm osc})` decreases `k_{\rm cut}` through the entropy
  redshifting factor
- if `g_* \simeq g_{*s}`, the net scaling is weak:
  \[
  k_{\rm cut}\propto g_*^{-1/12}
  \]
  so lower relativistic degrees of freedom give a modestly larger cutoff

That is why replacing the old fixed `g_* = g_{*s} = 100` approximation with
the lower physical values at MeV-scale `T_{\rm osc}` increased the benchmark
`k_{\rm cut}` and pushed `k_{\rm cross}` to larger wavenumber.

Important correction:

- the entropy factor is now
  \[
  \left(\frac{g_{s,0}}{g_{*s}(T_{\rm osc})}\right)^{1/3}
  \]
  not the inverse

This was a real bug in earlier versions.

### 8.2 PT cutoff ratio

\[
\frac{k_{\rm cut}^{\rm PT}}{k_{\rm cut}^{\rm noPT}}
=
\sqrt{\frac{t_{\rm osc}}{t_{\rm onset}^{\rm PT}}}.
\]

Then

\[
k_{\rm cut}^{\rm PT}
=
\left(\frac{k_{\rm cut}^{\rm PT}}{k_{\rm cut}^{\rm noPT}}\right)
k_{\rm cut}^{\rm noPT}.
\]

## 9. White-Noise Spectrum Normalization

This is the most important normalization point.

The script currently uses

\[
P_i(k)
=
6\pi^2\,
k_{\rm cut}^{-3}\,
\mathrm{var}\,
\Theta(k_{\rm cut}-k)
\]

for the no-growth-factor version, and

\[
P_i(k)
=
6\pi^2\,
k_{\rm cut}^{-3}\,
D_i(z_{\rm eq})^2\,
\mathrm{var}\,
\Theta(k_{\rm cut}-k)
\]

for the growth-factor-included version.

Important correction:

- the old prefactor `3/(4\pi)` has been removed
- the code now uses
  \[
  6\pi^2
  \]

This change was made because the previous prefactor undercounted the
white-noise amplitude.

In code:

- `WHITE_NOISE_PREFAC = 6.0 * math.pi**2`

## 10. Growth Factor

When the growth suppression is included, the script uses

\[
D_i(z_{\rm eq}) = \frac{1}{1+z_{\rm eq}},
\qquad
z_{\rm eq}=3402,
\]

so

\[
D_i(z_{\rm eq})^2
=
\frac{1}{(1+3402)^2}.
\]

In code:

- `DI2_EQ = 1.0 / ((1.0 + Z_EQ) ** 2)`

There are therefore two versions of the physical `P(k)` plots:

1. with `D_i^2`
2. without `D_i^2`

## 11. Dimensionless Power

The script defines

\[
\Delta^2(k)
=
\frac{k^3}{2\pi^2}P(k).
\]

This conversion is unit-invariant if:

- `k` is converted from `{\rm Mpc}^{-1}` to `{\rm kpc}^{-1}` by `10^{-3}`
- `P(k)` is converted from `{\rm Mpc}^3` to `{\rm kpc}^3` by `10^9`

The script now contains explicit unit sanity checks confirming this.

## 12. CDM Reference Used in the Current `P(k)` Plots

For the **current** `P(k)` plots, the script uses only the tabulated data:

- [paper_codes/Pk_CDM.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM.dat)

Conversion:

- first column:
  \[
  k[{\rm kpc}^{-1}] \to 1000\,k[{\rm Mpc}^{-1}]
  \]
- second column:
  \[
  P[{\rm kpc}^3] \to P/1000^3\ [{\rm Mpc}^3]
  \]

The current dat-only `P(k)` benchmark plots:

- do **not** show the EH linear continuation
- do **not** compute `k_{\rm cross}`
- only show the tabulated CDM spectrum in the plotted x-window

The current x/y window is:

\[
k \in [10^{-2}, 10^{1}]\,{\rm Mpc}^{-1},
\qquad
P(k)\in[10^{-3},10^{4}]\,{\rm Mpc}^3.
\]

## 13. Current Plot Outputs

The current physical `P(k)` outputs are:

- [Pk_physical_corrected.pdf](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/Pk_physical_corrected.pdf)
  - includes `D_i(z_{\rm eq})^2`
- [Pk_physical_noDi.pdf](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/Pk_physical_noDi.pdf)
  - does **not** include `D_i(z_{\rm eq})^2`

There are also `\Delta^2(k)` and kpc-versions in the script, but the current
paper-facing `P(k)` workflow is the dat-only Mpc-window pair above.

## 14. Benchmark Dependence

The benchmark shown in the current `P(k)` outputs is set by the plotting call.

At the time of writing this note, the current on-disk `P(k)` outputs use:

- `M_\phi = 10^{-20}` eV
- `H_*/M_\phi = 10^{-4}`
- `v_w = 0.5`
- PT endpoints `\beta/H_* = 4, 40`
- plus noPT

So the **file names stay fixed**, but the benchmark inside the plot can change
from run to run. The annotation inside the figure is the authoritative source
for which benchmark is currently on disk.

## 15. Current On-Disk Benchmark Values

These are the values printed by the most recent regeneration of:

- [Pk_physical_corrected.pdf](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/Pk_physical_corrected.pdf)
- [Pk_physical_noDi.pdf](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/Pk_physical_noDi.pdf)

for

\[
M_\phi = 10^{-20}\ {\rm eV},
\qquad
H_*/M_\phi = 10^{-4},
\qquad
v_w = 0.5.
\]

### 15.1 With `D_i(z_{\rm eq})^2`

- noPT:
  - `k_{\rm cut} = 26.5559\ {\rm Mpc}^{-1}`
  - `P_i = 5.48202\times10^{-10}\ {\rm Mpc}^3`
- `\beta/H_* = 4`:
  - `k_{\rm cut} = 0.243951\ {\rm Mpc}^{-1}`
  - `P_i = 2.45478\times10^{-4}\ {\rm Mpc}^3`
  - `P_0` ratio `= 4.47787641\times10^5`
- `\beta/H_* = 40`:
  - `k_{\rm cut} = 0.353729\ {\rm Mpc}^{-1}`
  - `P_i = 9.19613\times10^{-5}\ {\rm Mpc}^3`
  - `P_0` ratio `= 1.67750765\times10^5`

### 15.2 Without `D_i(z_{\rm eq})^2`

- noPT:
  - `P_i = 6.3484\times10^{-3}\ {\rm Mpc}^3`
- `\beta/H_* = 4`:
  - `P_i = 2.84274\times10^3\ {\rm Mpc}^3`
- `\beta/H_* = 40`:
  - `P_i = 1.06495\times10^3\ {\rm Mpc}^3`

These are the values that correspond to the current dat-only plotting window:

\[
k \in [10^{-2},10^{1}]\,{\rm Mpc}^{-1},
\qquad
P(k)\in[10^{-3},10^{4}]\,{\rm Mpc}^3.
\]

## 16. Summary of the Three Important Corrections

The current code includes these three normalization fixes:

1. Reduced Planck mass:
   \[
   \bar M_{\rm Pl} = 2.435\times10^{27}\ {\rm eV}
   \]
   not `1.22e28 eV`

2. Entropy factor:
   \[
   \left(\frac{g_{s,0}}{g_{s,*}}\right)^{1/3}
   \]
   not the inverse

3. White-noise prefactor:
   \[
   6\pi^2
   \]
   not `3/(4\pi)`

These are the key choices that control the absolute normalization of
`k_{\rm cut}` and `P_i(k)`.
