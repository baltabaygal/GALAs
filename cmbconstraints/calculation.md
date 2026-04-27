# Pre-Inflation CMB Isocurvature

This folder treats only the case where the axion field **did not** reach the
inflationary equilibrium distribution.

So the initial condition is a single background misalignment angle:

\[
\theta_0 = \text{fixed}.
\]

The inflationary fluctuation is

\[
\delta\theta = \frac{H_I}{2\pi f_\phi}.
\]

## Density Dependence

For the PT case, the axion abundance is modeled as

\[
\rho_\phi(\theta_0)
\propto
\xi(\theta_0;H_*,v_w,\beta/H_*)
f_{\rm anh}^{\rm noPT}(\theta_0)
(1-\cos\theta_0).
\]

The noPT reference is obtained by setting

\[
\xi = 1.
\]

The current noPT anharmonic factor is

\[
f_{\rm anh}^{\rm noPT}(\theta_0)
=
A_0
\left[1-\ln\cos^2\left(\frac{\theta_0}{2}\right)\right]^{\gamma_0},
\]

with

\[
A_0 = 0.373,
\qquad
\gamma_0 = 1.20.
\]

## Fractional Perturbation

The logarithmic response is

\[
\frac{\delta\rho_{\rm DM}}{\rho_{\rm DM}}
=
\left[
\frac{\partial\ln(1-\cos\theta_0)}{\partial\theta_0}
+
\frac{\partial\ln f_{\rm anh}}{\partial\theta_0}
+
\frac{\partial\ln \xi}{\partial\theta_0}
\right]\delta\theta.
\]

Define

\[
\mathcal D(\theta_0)
\equiv
\frac{\partial\ln(1-\cos\theta_0)}{\partial\theta_0}
+
\frac{\partial\ln f_{\rm anh}}{\partial\theta_0}
+
\frac{\partial\ln \xi}{\partial\theta_0}.
\]

Then

\[
P_S(k_*)
=
\left[
\frac{H_I}{2\pi f_\phi}
\mathcal D(\theta_0)
\right]^2.
\]

## Explicit Derivatives

### Potential term

\[
\frac{\partial\ln(1-\cos\theta_0)}{\partial\theta_0}
=
\frac{\sin\theta_0}{1-\cos\theta_0}
=
\cot\left(\frac{\theta_0}{2}\right).
\]

### Anharmonic term

Let

\[
h(\theta_0)
=
1-\ln\cos^2\left(\frac{\theta_0}{2}\right).
\]

Then

\[
f_{\rm anh}^{\rm noPT} = A_0 h^{\gamma_0},
\]

so

\[
\frac{\partial\ln f_{\rm anh}}{\partial\theta_0}
=
\gamma_0\frac{1}{h}\frac{dh}{d\theta_0}.
\]

Since

\[
\frac{dh}{d\theta_0}
=
\tan\left(\frac{\theta_0}{2}\right),
\]

we get

\[
\frac{\partial\ln f_{\rm anh}}{\partial\theta_0}
=
\gamma_0
\frac{\tan(\theta_0/2)}
{1-\ln\cos^2(\theta_0/2)}.
\]

### PT term

For the current `xi_model`, the derivative is evaluated numerically:

\[
\frac{\partial\ln \xi}{\partial\theta_0}
\approx
\frac{\ln\xi(\theta_0+\Delta\theta)-\ln\xi(\theta_0-\Delta\theta)}
{2\Delta\theta}.
\]

The code uses a centered finite difference with clipping near the physical
boundaries `0` and `\pi`.

## Current Model Ingredients Included

When evaluating `\xi(\theta_0)`, the current central `xi_model` includes:

- `A_0(\theta_0)`
- `\lambda(\theta_0)`
- pilot `\kappa(\theta_0,v_w,H_*)`
- the compact homogeneous baseline `\xi_{\rm DM}`
- the BM geometry response

So the PT derivative is the derivative of the **full deployed central model**,
not of a reduced toy ansatz.

## What This Folder Does Not Yet Do

This folder does **not** yet treat the equilibrium/stochastic initial
distribution:

\[
P_{\rm eq}(\theta)\propto
\exp\left[-\frac{8\pi^2}{3H_I^4}V(\theta)\right].
\]

That case needs a separate angle average over the inflationary ensemble and
will be handled later.
