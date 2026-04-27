# Isocurvature Status Note

This note describes exactly what the current implementation in
[run_isocurvature_white_noise.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/run_isocurvature_white_noise.py)
is doing, what we changed during debugging, and what the current outputs mean.

The point is to make the current calculation auditable.

## Current code path

The scan uses:

- `H_LIST = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]`
- `VW_LIST = [0.3, 0.5, 0.7, 0.9]`
- `BETA_LIST = [4.0, 8.0, 12.0, 20.0, 40.0]`
- `THETA_GRID = linspace(0.01, pi-0.01, 500)`

It loads:

- `xi_model` via `load_default_model()`
- `P_CDM(k)` from [Pk_CDM.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM.dat)

## No-PT baseline used in the code

The angular density baseline is

\[
\rho_{\rm noPT}(\theta)
=
(1-\cos\theta)\,f_{\rm anh}^{\rm noPT}(\theta)
\]

with

\[
f_{\rm anh}^{\rm noPT}(\theta)
=
\left[1-\ln\!\big(\cos^2(\theta/2)\big)\right]^{2.216}.
\]

In code:

- `rho_common = (1 - cos(theta)) * fanh_no_pt(theta)`

The no-PT contrast variance is

\[
\langle \delta^2 \rangle_{\rm noPT}
=
\frac{1}{\pi}\int_0^\pi
\left(
\frac{\rho_{\rm noPT}(\theta)-\bar\rho_{\rm noPT}}{\bar\rho_{\rm noPT}}
\right)^2
d\theta.
\]

In code this is:

- `delta2_no_pt = delta2_from_rho(rho_common, THETA_GRID)`

with

\[
\bar\rho=\frac{1}{\pi}\int_0^\pi \rho(\theta)\,d\theta.
\]

## PT density used currently

### Original version

Originally the script used

\[
\rho_{\rm PT}(\theta)=\xi(\theta)\,\rho_{\rm noPT}(\theta).
\]

This produced a problem: the compact `xi_model` does not approach pointwise
`\xi \to 1` in the fast-PT limit. Instead it approaches a nontrivial fast-PT
baseline.

That meant the script was **not** comparing PT to the actual no-PT limit.

### Current version

The current script normalizes `\xi` by a large-`\beta/H_*` reference:

\[
\xi_{\rm rel}(\theta;H_*,v_w,\beta/H_*)
=
\frac{\xi(\theta;H_*,v_w,\beta/H_*)}
{\xi(\theta;H_*,v_w,\beta/H_*=10^6)}.
\]

Then it uses

\[
\rho_{\rm PT}(\theta)
=
\xi_{\rm rel}(\theta)\,\rho_{\rm noPT}(\theta).
\]

In code:

- `xi_vals = model._eval_core(... beta_over_h=beta)["xi"]`
- `xi_fast_vals = model._eval_core(... beta_over_h=1e6)["xi"]`
- `xi_rel = xi_vals / xi_fast_vals`
- `rho_pt = xi_rel * rho_common`

So yes: the current code **is normalizing the PT response by the model's own
fast-PT asymptote**.

This was introduced specifically because the raw compact `xi_model` does not
return to the no-PT limit by itself.

## PT variance used currently

The PT contrast variance is

\[
\langle \delta^2 \rangle_{\rm PT}
=
\frac{1}{\pi}\int_0^\pi
\left(
\frac{\rho_{\rm PT}(\theta)-\bar\rho_{\rm PT}}{\bar\rho_{\rm PT}}
\right)^2
d\theta.
\]

In code:

- `delta2_pt = delta2_from_rho(rho_pt, THETA_GRID)`

Important:

- this is a **fractional contrast** variance
- any overall multiplicative rescaling of `\rho_{\rm PT}` drops out
- so this is a same-abundance style contrast measure, not an absolute density variance

## Oscillation / cutoff time currently used

The no-PT oscillation time is hard-coded as

\[
t_{\rm osc}^{\rm noPT}=1.5.
\]

In code:

- `TOSC_NO_PT = 1.5`

The PT percolation time is taken from `xi_model`:

\[
t_p^{\rm PT}
=
\texttt{model.\_eval\_core(theta0=1.0, ...)[\"tp\"]}.
\]

### Correction that was made

Originally the script used raw `t_p^{\rm PT}` in the `k_{\rm cut}` / `P_0`
factor, even when `t_p < t_{\rm osc}`.

That was wrong for your physical interpretation.

The current script uses

\[
t_{\rm eff}^{\rm PT}=\max(t_{\rm osc}^{\rm noPT},\,t_p^{\rm PT}).
\]

So:

- if `t_p > t_{\rm osc}`, PT delays onset
- if `t_p < t_{\rm osc}`, PT does **not** start oscillations earlier than no-PT

In code:

- `t_eff_pt = max(tp_no_pt, tp_pt)`

## White-noise amplitude used currently

The current script uses

\[
\frac{P_0^{\rm PT}}{P_0^{\rm noPT}}
=
\frac{\langle \delta^2 \rangle_{\rm PT}}{\langle \delta^2 \rangle_{\rm noPT}}
\left(
\frac{t_{\rm eff}^{\rm PT}}{t_{\rm osc}^{\rm noPT}}
\right)^{3/2}.
\]

In code:

- `p0_ratio = (delta2_pt / delta2_no_pt) * (t_eff_pt / tp_no_pt)**1.5`

This means:

- if `t_p < t_{\rm osc}`, the time factor is exactly `1`
- then `P_0` is controlled only by the contrast ratio

## k_cut ratio used currently

The script reports

\[
\frac{k_{\rm cut}^{\rm PT}}{k_{\rm cut}^{\rm noPT}}
=
\sqrt{\frac{t_{\rm osc}^{\rm noPT}}{t_{\rm eff}^{\rm PT}}}.
\]

In code:

- `k_cut_ratio_pt_to_no_pt = (tp_no_pt / t_eff_pt)**0.5`

So for `t_p < t_{\rm osc}`, the current code gives:

\[
\frac{k_{\rm cut}^{\rm PT}}{k_{\rm cut}^{\rm noPT}}=1.
\]

## k_eq convention used currently

Absolute normalization is not fixed. The code defines:

- `k_ref = 1 Mpc^{-1}`
- `P_0^{\rm noPT} = P_{\rm CDM}(k_ref)`
- `P_0^{\rm PT} = (P_0^{\rm PT}/P_0^{\rm noPT}) P_{\rm CDM}(k_ref)`

Then it inverts the decreasing CDM tail to find the relative intersection:

\[
k_{\rm eq,iso}^{\rm noPT}\equiv 1\ {\rm Mpc}^{-1},
\qquad
\frac{k_{\rm eq,iso}^{\rm PT}}{k_{\rm eq,iso}^{\rm noPT}}
=
\frac{k(P_{\rm CDM}=P_0^{\rm PT})}{1\ {\rm Mpc}^{-1}}.
\]

## Current benchmark results

From
[outputs/summary.json](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs/summary.json):

### `H_*=0.05`, `v_w=0.5`

- `beta/H_*=4`
  - `P_0^{PT}/P_0^{noPT} = 83.5293`
  - `k_eq^{PT}/k_eq^{noPT} = 0.12159`
  - `k_cut^{PT}/k_cut^{noPT} = 0.20541`

- `beta/H_*=40`
  - `P_0^{PT}/P_0^{noPT} = 38.4773`
  - `k_eq^{PT}/k_eq^{noPT} = 0.18721`
  - `k_cut^{PT}/k_cut^{noPT} = 0.29785`

### `H_*=2.0`, `v_w=0.5`

- `beta/H_*=4`
  - `P_0^{PT}/P_0^{noPT} = 0.86871`
  - `k_eq^{PT}/k_eq^{noPT} = 1.06123`
  - `k_cut^{PT}/k_cut^{noPT} = 1.0`

- `beta/H_*=40`
  - `P_0^{PT}/P_0^{noPT} = 0.96356`
  - `k_eq^{PT}/k_eq^{noPT} = 1.01581`
  - `k_cut^{PT}/k_cut^{noPT} = 1.0`

### Additional scan values that matter

For `v_w=0.5`, the current `P_0` ratios are:

#### `H_*=0.5`

- `beta/H_*=4`: `2.6042`
- `8`: `2.0193`
- `12`: `1.6893`
- `20`: `1.3577`
- `40`: `1.0447`

#### `H_*=1.0`

- `beta/H_*=4`: `1.0022`
- `8`: `0.8130`
- `12`: `0.8419`
- `20`: `0.8798`
- `40`: `0.9225`

#### `H_*=2.0`

- `beta/H_*=4`: `0.8687`
- `8`: `0.8976`
- `12`: `0.9178`
- `20`: `0.9406`
- `40`: `0.9636`

## What is likely wrong / at least questionable

There are two obvious places to question:

### 1. The `xi_rel = xi / xi_fast` normalization

This is the most nontrivial intervention currently in the script.

It was added because the compact `xi_model` does not approach pointwise no-PT
at large `\beta/H_*`. So the code now forces a relative PT response by dividing
out the fast-PT asymptote.

If this is not the right object physically, then the current `P_0` ratio is not
the quantity you want.

### 2. Using fractional contrast variance rather than absolute variance

The current script uses:

\[
\delta(\theta)=\frac{\rho(\theta)-\bar\rho}{\bar\rho}.
\]

So any overall enhancement of the abundance is removed.

If the physical white-noise amplitude you want is tied to absolute density
variance instead, then the current calculation is suppressing that effect by construction.

## Minimal audit questions

To debug the physics, the clean questions are:

1. Should the PT density be
   - `rho_PT = xi * rho_noPT`
   or
   - `rho_PT = (xi / xi_fast) * rho_noPT`
   ?

2. Should the white-noise amplitude use
   - fractional variance `Var(rho)/bar(rho)^2`
   or
   - absolute variance `Var(rho)`?

3. Is the no-PT comparison intended to be
   - true no-PT standard misalignment,
   - fast-PT limit of the compact model,
   - or abundance-matched after retuning the overall relic density?

Until those are fixed, the current plots should be read as:

- **fractional-contrast white noise**
- with **effective onset time** `t_eff=max(t_osc,t_p)`
- and **PT response normalized by the model's own fast-PT asymptote**

## Deprecated comparison plot convention

There was an older `Pk_comparison` construction used during debugging that
should not be interpreted as the physical white-noise power spectrum.

That deprecated construction defined

\[
P_0^{\rm noPT} = P_{\rm CDM}(k_{\rm ref}),
\qquad
k_{\rm ref}=1\ {\rm Mpc}^{-1},
\]

and then set

\[
P_0^{\rm PT}
=
\left(\frac{P_0^{\rm PT}}{P_0^{\rm noPT}}\right) P_{\rm CDM}(k_{\rm ref}).
\]

So the horizontal PT/noPT lines in that plot were anchored to the CDM curve by
construction. That makes them useful only as a relative intersection diagnostic.

It is **not** the physically normalized white-noise spectrum

\[
P_i(k)=\frac{3}{4\pi}k_{\rm cut}^{-3}\,\mathrm{var}\,\Theta(k_{\rm cut}-k),
\]

which depends on the chosen `M_\phi` through `k_{\rm cut}`.

Therefore:

- old high horizontal lines in the earlier `Pk_comparison` figure are expected
  from the diagnostic normalization
- they should not be compared directly to the later physically normalized
  `P(k)` plots
- the physical plots are the ones generated by
  [plot_isocurvature.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/plot_isocurvature.py)
  with explicit `M_\phi`
