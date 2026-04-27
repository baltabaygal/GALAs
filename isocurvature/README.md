# Isocurvature White-Noise Scan

This folder computes the PT modification of the post-inflationary axion
isocurvature white-noise amplitude using the accepted `xi_model` package and
the tabulated CDM power spectrum in:

- [../Pk_CDM.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM.dat)

Outputs are written to:

- [outputs](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/outputs)

## Definitions

For a uniform post-inflationary `\theta_0` distribution on `[0,\pi]`:

\[
\rho_{\rm PT}(\theta_0)=\xi_{\rm rel}(\theta_0)\,(1-\cos\theta_0)\,f_{\rm anh}^{\rm noPT}(\theta_0)
\]

\[
\rho_{\rm noPT}(\theta_0)=(1-\cos\theta_0)\,f_{\rm anh}^{\rm noPT}(\theta_0)
\]

with the Eq. 17-style no-PT anharmonic factor used in this calculation:

\[
f_{\rm anh}^{\rm noPT}(\theta_0)=\left[1-\ln\!\big(\cos^2(\theta_0/2)\big)\right]^{2.216}.
\]

The relative PT enhancement is defined by normalizing the model response to its
own fast-PT asymptote:

\[
\xi_{\rm rel}(\theta_0;H_*,v_w,\beta/H_*)
=
\frac{\xi(\theta_0;H_*,v_w,\beta/H_*)}
{\xi(\theta_0;H_*,v_w,\beta/H_*\to\infty)}.
\]

In the script, the asymptotic reference is evaluated at
`beta/H_* = 10^6`, which is numerically in the fast-PT limit for this model.

The white-noise amplitude ratio is

\[
\frac{P_0^{\rm PT}}{P_0^{\rm noPT}}
=
\frac{\langle |\delta|^2\rangle_{\rm PT}}{\langle |\delta|^2\rangle_{\rm noPT}}
\left(\frac{t_{\rm eff}^{\rm PT}}{t_{\rm osc}^{\rm noPT}}\right)^{3/2},
\]

with the no-PT reference
\[
t_{\rm osc}^{\rm noPT}=1.5\,M_\phi^{-1}.
\]

The PT onset used for the cutoff/amplitude factor is
\[
t_{\rm eff}^{\rm PT}=\max\!\big(t_{\rm osc}^{\rm noPT},\, t_p^{\rm PT}\big),
\]
so a phase transition with `t_p<t_{\rm osc}` does not start oscillations earlier than
the no-PT case; it only modifies the density response through `\xi(\theta_0)`.

## `k_eq,iso` convention

The absolute white-noise normalization is not fixed by the available inputs, so
the intersection with the CDM spectrum is treated relatively.

This scan uses:

- `k_ref = 1\,{\rm Mpc}^{-1}` on the decreasing CDM tail
- `P_0^{\rm noPT}` normalized to `P_{\rm CDM}(k_ref)`
- `P_0^{\rm PT} = (P_0^{\rm PT}/P_0^{\rm noPT})\,P_{\rm CDM}(k_ref)`

and the relative cutoff shift
\[
\frac{k_{\rm cut}^{\rm PT}}{k_{\rm cut}^{\rm noPT}}
=
\sqrt{\frac{t_{\rm osc}^{\rm noPT}}{t_{\rm eff}^{\rm PT}}}.
\]

Then

\[
\frac{k_{\rm eq,iso}^{\rm PT}}{k_{\rm eq,iso}^{\rm noPT}}
=
\frac{k(P_{\rm CDM}=P_0^{\rm PT})}{k_{\rm ref}},
\qquad
k_{\rm eq,iso}^{\rm noPT}\equiv k_{\rm ref}.
\]

So the reported `k_eq` shift is a reproducible relative shift on the CDM tail,
not a claim of an absolutely normalized physical `k_eq,iso`.
