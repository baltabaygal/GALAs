# `k_cross` Calculation Note

This note records the **explicit current computation** of the crossing scale
`k_cross`, defined by

\[
P_i(k_{\rm cross}) = P_{\rm CDM}(k_{\rm cross}).
\]

It is written against the current implementation in
[plot_isocurvature.py](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/isocurvature/plot_isocurvature.py),
using the current tabulated CDM spectrum
[Pk_CDM 2.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM%202.dat).

## Definition

For the white-noise isocurvature spectrum used in the code,

\[
P_i(k)
=
6\pi^2\,
k_{\rm cut}^{-3}\,
\frac{1}{D_i(z_{\rm eq})^2}\,
\mathrm{var}\,
\Theta(k_{\rm cut}-k).
\]

Below the cutoff, `P_i(k)` is constant, so the crossing condition is just

\[
P_{\rm CDM}(k_{\rm cross}) = P_i^{(0)},
\]

where

\[
P_i^{(0)}
=
6\pi^2\,
k_{\rm cut}^{-3}\,
\frac{1}{D_i(z_{\rm eq})^2}\,
\mathrm{var}.
\]

## Inputs Used Here

This explicit example is for:

- `M_\phi = 10^{-16}\ \mathrm{eV}`
- noPT
- harmonic reference variance
  \[
  \mathrm{var} = \frac{4}{5} = 0.8
  \]

Current constants in code:

- white-noise prefactor
  \[
  6\pi^2 = 59.2176264065
  \]
- growth factor convention
  \[
  D_i(z_{\rm eq}) = 3.7\times 10^{-4}
  \]
- therefore
  \[
  D_i(z_{\rm eq})^2 = 1.369\times 10^{-7}
  \]
  and
  \[
  \frac{1}{D_i(z_{\rm eq})^2} = 7.304601899\times 10^6
  \]
- reduced Hubble parameter
  \[
  h = 0.674
  \]

## Step 1: Compute `k_cut`

For this mass, the current code now uses the temperature-dependent
`g_*(T_{\rm osc})`, `g_{*s}(T_{\rm osc})` correction in the physical cutoff
conversion. With that correction, it gives

\[
k_{\rm cut}^{\rm noPT}
\propto
g_*(T_{\rm osc})^{1/4}\,g_{*s}(T_{\rm osc})^{-1/3},
\]

so using the lower physical MeV-scale degrees of freedom instead of the old
fixed `100` approximation increases `k_{\rm cut}` slightly and therefore
lowers the flat white-noise level `P_i^{(0)} \propto k_{\rm cut}^{-3}`.
That is exactly why the crossing moves to larger `k`.

\[
k_{\rm cut} = 3349.816022841183\ {\rm Mpc}^{-1}.
\]

Converted to `h\,\mathrm{Mpc}^{-1}` using

\[
k[h\,{\rm Mpc}^{-1}] = \frac{k[{\rm Mpc}^{-1}]}{h},
\]

this is

\[
k_{\rm cut} = \frac{3349.816022841183}{0.674}
= 4970.053446351903\ h\,{\rm Mpc}^{-1}.
\]

## Step 2: Compute the White-Noise Level

Using

\[
P_i^{(0)}
=
6\pi^2\,
k_{\rm cut}^{-3}\,
\frac{1}{D_i(z_{\rm eq})^2}\,
\mathrm{var},
\]

with

- `k_cut = 3349.816022841183`
- `var = 0.8`

the code gives

\[
P_i^{(0)} = 0.009206078376381068\ {\rm Mpc}^3.
\]

Converted to `h^{-3}\mathrm{Mpc}^3` using

\[
P[h^{-3}{\rm Mpc}^3] = h^3 P[{\rm Mpc}^3],
\]

this becomes

\[
P_i^{(0)} = 0.009206078376381068 \times 0.674^3
= 0.002819074932584878\ h^{-3}{\rm Mpc}^3.
\]

## Step 3: Find the Bracketing CDM Points

The code interpolates the tabulated
[Pk_CDM 2.dat](/home/gala/Desktop/AXION_PT/CODES/STD_FANH/paper_codes/Pk_CDM%202.dat)
in log-log space.

For the crossing, the first sign change now occurs between:

\[
k_0 = 32.7363\ {\rm Mpc}^{-1}
\]
\[
k_1 = 33.9674\ {\rm Mpc}^{-1}
\]

with corresponding CDM powers

\[
P_{\rm CDM}(k_0) = 0.0100777\ {\rm Mpc}^3
\]
\[
P_{\rm CDM}(k_1) = 0.00911699\ {\rm Mpc}^3.
\]

Define

\[
f(k) = P_i^{(0)} - P_{\rm CDM}(k).
\]

Then at the two bracketing points,

\[
f(k_0) = -8.71621623618932\times 10^{-4}
\]
\[
f(k_1) = 8.9088376381068\times 10^{-5}.
\]

So the crossing lies between `k_0` and `k_1`, as expected.

## Step 4: Linear Interpolation in the Bracket

The current code then solves for the zero of `f(k)` by linear interpolation
inside that tiny bracket:

\[
k_{\rm cross}

=
k_0 + \frac{-f(k_0)}{f(k_1)-f(k_0)} (k_1-k_0).
\]

Numerically this gives

\[
k_{\rm cross} = 33.85323786973933\ {\rm Mpc}^{-1}.
\]

Converted to `h\,\mathrm{Mpc}^{-1}`:

\[
k_{\rm cross}
=
\frac{33.85323786973933}{0.674}
= 50.22735588982096\ h\,{\rm Mpc}^{-1}.
\]

At this point the interpolated CDM spectrum is

\[
P_{\rm CDM}(k_{\rm cross})
= 0.009206078376381068\ {\rm Mpc}^3,
\]

which matches the white-noise level

\[
P_i^{(0)} = 0.009206078376381068\ {\rm Mpc}^3
\]

to the expected interpolation accuracy.

## Final Result

For the current code path, with

- `M_\phi = 10^{-16}\ \mathrm{eV}`
- noPT
- `\mathrm{var} = 4/5`

the crossing is:

\[
k_{\rm cross} = 26.1682\ {\rm Mpc}^{-1}
\]

or equivalently

\[
k_{\rm cross} = 38.8252\ h\,{\rm Mpc}^{-1}.
\]

## Why This May Differ From Another Number

If another estimate gives a value closer to `48 h\,\mathrm{Mpc}^{-1}`, the
difference is **not** coming from the `h` conversion used here. The conversion
used in the code is simply

\[
k[h\,{\rm Mpc}^{-1}] = \frac{k[{\rm Mpc}^{-1}]}{h},
\qquad h=0.674.
\]

So a discrepancy at the level of `38.8` vs `48` must come from one of the
inputs instead:

- a different `P_{\rm CDM}` table
- a different `D_i(z_{\rm eq})`
- a different `k_{\rm cut}`
- a different variance normalization

not from the `h` conversion itself.
