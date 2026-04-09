# Micropuncture Angiogenesis Model

Phase-field PDE model for micropuncture-induced angiogenesis, coupling vessel evolution (Allen-Cahn) with VEGF diffusion-reaction. Implemented in FEniCS.

## What this does

Simulates how precision micropuncture accelerates blood vessel growth in collagen scaffolds. The model takes fluorescence microscopy images from experiment as initial conditions and evolves a coupled system:

- **phi(x,t)**: phase field representing vessel density (0=tissue, 1=vessel), governed by Allen-Cahn with a double-well potential
- **c(x,t)**: VEGF concentration, governed by reaction-diffusion with optional localized sources at puncture sites

Control group (no micropuncture) should show minimal vessel change; treatment group introduces a Gaussian VEGF source to drive sprouting.

## Model equations

```
d_t(c) = D_c * div(phi * grad(c)) - mu_c * phi * c + alpha(x) * phi
d_t(phi) = eps^2 * laplacian(phi) - 8*phi*(1-phi)*(1-2*phi) + beta_c * c
```

where alpha(x) is a Gaussian VEGF source at the micropuncture site (zero for control).

## Structure

```
src/
  convert_grayscale.py       - image preprocessing: .jpg -> *_raw.npy
  micropuncture_model.py     - main FEniCS solver (Allen-Cahn + VEGF)
gray_scales/
  (place *_raw.npy files here, or generate with convert_grayscale.py)
fig/                          - output: phi and c field plots
res/                          - output: saved field vectors
```

## Usage

Requires FEniCS (legacy, 2019.x), numpy, scipy, matplotlib.

```bash
# step 1: convert microscopy images to grayscale arrays
python src/convert_grayscale.py --source_folder /path/to/microscopy/images

# step 2a: control group (no VEGF source)
python src/micropuncture_model.py --group control --Nx 128 --Nt 400 --save_init

# step 2b: treatment group (Gaussian VEGF source at center)
python src/micropuncture_model.py --group treatment --Nx 128 --Nt 400 --alpha_strength 1.0 --save_init
```

Key parameters: `--eps` (interface width), `--Dc` (VEGF diffusion), `--beta_c` (coupling strength), `--phi_pct` (vessel threshold percentile). Use `--mobility_gate` to enable concentration-dependent mobility and `--mass_stab` for phi mass conservation.

Run `python src/micropuncture_model.py --help` for all options.

## Status

Work in progress. Control group simulations are being validated for steady-state stability. Next steps: full treatment group comparison against experimental data, and parameter calibration.

## References

- Hancock et al., "Induction of scaffold angiogenesis by recipient vasculature precision micropuncture," *Microvascular Research*, 2021.
- Horchler et al., "Vascular persistence following precision micropuncture," *Microcirculation*, 2024.

## Context

Undergraduate research supervised by Prof. Wenrui Hao, Penn State Mathematics, with advising and feedback from Dr. Boyi Wang and Dr. Shun Wang. This work received the Norman Freed Undergraduate Research Award at the Eberly College of Science Undergraduate Research Exhibition.
