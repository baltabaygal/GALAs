# `k_cross` Scaling Fit

This note gives a compact analytic approximation for the crossing scale
`k_cross`, defined by

\[
P_i(k_{\rm cross}) = P_{\rm CDM}(k_{\rm cross}),
\]

using the **current code conventions**, including the corrected
temperature-dependent `g_*(T_{\rm osc})`, `g_{*s}(T_{\rm osc})` treatment in
the physical cutoff.

It is written against:

- [plot_isocurvature.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/plot_isocurvature.py)
- [Pk_CDM 2.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM%202.dat)

## 1. White-Noise Side

The code uses

\[
P_i^{(0)} = 6\pi^2\,k_{\rm cut}^{-3}\,D_i(z_{\rm eq})^{-2}\,\mathrm{var}.
\]

For the physical cutoff,

\[
k_{\rm cut}^{\rm noPT}
=
H_{\rm osc}
\left(\frac{T_0}{T_{\rm osc}}\right)
\left(\frac{g_{s,0}}{g_{*s}(T_{\rm osc})}\right)^{1/3},
\qquad
H_{\rm osc}=\frac{M_\phi}{3},
\]

with `T_osc` determined self-consistently from

\[
T_{\rm osc}
=
\sqrt{H_{\rm osc}\bar M_{\rm Pl}}
\left(\frac{90}{\pi^2 g_*(T_{\rm osc})}\right)^{1/4}.
\]

So parametrically,

\[
k_{\rm cut}^{\rm noPT}
\propto
M_\phi^{1/2}\,
g_*(T_{\rm osc})^{1/4}\,
g_{*s}(T_{\rm osc})^{-1/3}.
\]

For PT cases,

\[
k_{\rm cut}^{\rm PT}
=
k_{\rm cut}^{\rm noPT}
\sqrt{\frac{t_{\rm osc}}{t_{\rm onset}}},
\qquad
t_{\rm onset}=\max(t_p,t_{\rm osc}).
\]

Therefore

\[
P_i^{(0)}
\propto
\mathrm{var}\,
M_\phi^{-3/2}\,
\left(\frac{t_{\rm onset}}{t_{\rm osc}}\right)^{3/2}
g_*(T_{\rm osc})^{-3/4}
g_{*s}(T_{\rm osc} )^{+1}.
\]

## 2. CDM Side

In the currently relevant crossing region, the tabulated CDM spectrum is well
approximated by a power law in `h` units.

Fitting [Pk_CDM 2.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM%202.dat)
over

\[
5 \le k \le 100\ h\,{\rm Mpc}^{-1}
\]

gives

\[
P_{\rm CDM}(k)\simeq A_{\rm CDM}\,k^{-\nu},
\]

with

\[
\nu = 2.66267913,
\qquad
A_{\rm CDM} = 95.1424.
\]

## 3. Resulting `k_cross` Scaling

From `P_i^{(0)} = P_{\rm CDM}(k_{\rm cross})`,

\[
k_{\rm cross}
\propto
M_\phi^{3/(2\nu)}
\mathrm{var}^{-1/\nu}
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{3/(2\nu)}
g_*(T_{\rm osc})^{3/(4\nu)}
g_{*s}(T_{\rm osc})^{-1/\nu}.
\]

With the fitted `\nu`,

\[
\frac{1}{\nu}=0.37556,
\qquad
\frac{3}{2\nu}=0.56334,
\qquad
\frac{3}{4\nu}=0.28167.
\]

So the compact scaling law is

\[
k_{\rm cross}
\propto
M_\phi^{0.563}\,
\mathrm{var}^{-0.376}\,
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.563}\,
g_*(T_{\rm osc})^{0.282}\,
g_{*s}(T_{\rm osc})^{-0.376}.
\]

Normalizing to the current noPT benchmark at `M_\phi = 10^{-18}` eV gives

\[
k_{\rm cross}
\simeq
2.49\,
\left(\frac{M_\phi}{10^{-18}\ {\rm eV}}\right)^{0.563}
\left(\frac{\mathrm{var}}{2.0077}\right)^{-0.376}
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.563}
\left(\frac{g_*(T_{\rm osc})}{3.36}\right)^{0.282}
\left(\frac{g_{*s}(T_{\rm osc})}{3.91}\right)^{-0.376}
\ h\,{\rm Mpc}^{-1}.
\]

This is the cleanest current formula.

## 4. Simplified Version

If you want something closer to the intuitive scaling, you can round it to

\[
k_{\rm cross}
\sim
\left(\frac{M_\phi}{10^{-18}\ {\rm eV}}\right)^{0.6}
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.6}
\times
\text{(variance factor)}
\times
\text{(mild }g_*,g_{*s}\text{ factor)}.
\]

That is qualitatively right, but the explicit fitted expression above is more
accurate for the current CDM table.

## 5. Spot Checks

### Reference noPT, `M_\phi = 10^{-18}` eV

- numeric:
  \[
  k_{\rm cross}=2.4866\ h\,{\rm Mpc}^{-1}
  \]
- fitted formula:
  \[
  2.4866\ h\,{\rm Mpc}^{-1}
  \]

### Harmonic reference, `M_\phi = 10^{-18}` eV, `\mathrm{var}=4/5`

- numeric:
  \[
  k_{\rm cross}=3.6079\ h\,{\rm Mpc}^{-1}
  \]
- fitted formula:
  \[
  3.5131\ h\,{\rm Mpc}^{-1}
  \]
- relative error:
  \[
  -2.6\%
  \]

### Harmonic reference, `M_\phi = 10^{-16}` eV, `\mathrm{var}=4/5`

- numeric:
  \[
  k_{\rm cross}=50.2274\ h\,{\rm Mpc}^{-1}
  \]
- fitted formula:
  \[
  47.0297\ h\,{\rm Mpc}^{-1}
  \]
- relative error:
  \[
  -6.4\%
  \]

So the current fitted expression is already good at the few-percent level for
the benchmark cases, and it captures exactly the structure you wanted:

- mass scaling
- onset-time scaling
- explicit thermodynamic `g_*`, `g_{*s}` dependence

## 6. Where the Error Comes From

The dominant missing ingredient in the simplified `k_cross` scaling is the
PT-dependent variance, not the thermodynamic `g_* , g_{*s}` factors.

Using the completed `k_cross` validation set currently on disk:

- [k_cross_formula_test.csv](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/k_cross_formula_test.csv)
- [k_cross_formula_test_summary.json](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/k_cross_formula_test_summary.json)

the following comparisons hold.

### Simplified scaling only

\[
k_{\rm cross}\propto
\left(\frac{M_\phi}{10^{-18}\ {\rm eV}}\right)^{0.6}
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.6}
\]

On the tested set:

- mean absolute fractional error: `17.7%`
- median absolute fractional error: `18.3%`
- 90th percentile absolute fractional error: `33.5%`

### Add variance, still omit `g_* , g_{*s}`

\[
k_{\rm cross}\propto
\left(\frac{M_\phi}{10^{-18}\ {\rm eV}}\right)^{0.6}
\left(\frac{\mathrm{var}}{2.0077}\right)^{-0.37556}
\left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.6}
\]

On the same tested set:

- mean absolute fractional error: `6.6%`
- median absolute fractional error: `8.4%`
- 90th percentile absolute fractional error: `10.4%`

So the variance factor is what collapses the error from order `20%` to order
`10%`.

### Why `g_* , g_{*s}` do not help on the current test set

The completed validation set currently on disk is only for a single physical
mass:

\[
M_\phi = 10^{-16}\ {\rm eV}.
\]

Therefore `T_{\rm osc}`, `g_*(T_{\rm osc})`, and `g_{*s}(T_{\rm osc})` are
constant across the whole tested set. Since they do not vary from point to
point, they cannot explain the residual scatter there.

So on the current validation set:

- adding `g_* , g_{*s}` does **not** reduce the spread
- the spread is controlled by the PT-dependent variance

The thermodynamic factors will matter only once the validation is extended to
multiple masses.

## 7. Variance Should Be Modeled as a Separate PT Factor

The scan also shows that the variance is not a function of `t_{\rm onset}`
alone.

In particular:

- for `H_* \gtrsim 1.5`, many cases are clamped to
  \[
  t_{\rm onset}=t_{\rm osc}=1.5
  \]
  but the variance still changes significantly across the PT parameter scan

So the right reduced object to model is

\[
R_{\rm var}(H_*,x)
\equiv
\frac{\mathrm{var}_{\rm PT}}{\mathrm{var}_{\rm noPT}(H_*)},
\qquad
x\equiv \frac{t_{\rm osc}}{t_{\rm onset}}.
\]

This removes the large noPT baseline trend with `H_*` and isolates the PT
correction.

A minimal physically sensible ansatz is

\[
R_{\rm var}(H_*,x)
=
R_\infty(H_*)
+
\bigl[1-R_\infty(H_*)\bigr]x^p,
\qquad
x=\frac{t_{\rm osc}}{t_{\rm onset}}.
\]

Interpretation:

- `x \to 0`: strong PT delay
- `x \to 1`: fast/clamped branch
- `R_\infty(H_*)` captures the residual non-universality of the `x=1` branch

So the clean analytic program for `k_{\rm cross}` is:

1. model `R_{\rm var}(H_*,x)`
2. plug it into
   \[
   k_{\rm cross}\propto
   M_\phi^{0.6}
   \left(\frac{t_{\rm osc}}{t_{\rm onset}}\right)^{0.6}
   \mathrm{var}^{-\alpha}
   \]
3. treat `g_* , g_{*s}` as secondary corrections to be validated in a
   multi-mass scan
