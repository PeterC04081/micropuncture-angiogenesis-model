# micropuncture_model.py
# Phase-field + VEGF coupled PDE model for micropuncture-induced angiogenesis
# Solves Allen-Cahn (phi) + reaction-diffusion (c) on [0,1]^2 using FEniCS
#
# phi = vessel phase field (0=tissue, 1=vessel)
# c = VEGF concentration
# control group: no external source (alpha=0, c_init=0)
# treatment group: localized Gaussian VEGF source at puncture site

import os
import argparse
import numpy as np

from fenics import *
from ufl import min_value, tanh

# --- command line args ---
parser = argparse.ArgumentParser(description="Micropuncture vessel growth FEniCS model")
parser.add_argument("--group", type=str, default="control", choices=["control", "treatment"],
                    help="control (no source) or treatment (localized VEGF source)")
parser.add_argument("--gray_folder", type=str, default="gray_scales",
                    help="folder containing *_raw.npy grayscale ROI files")
parser.add_argument("--Nx", type=int, default=128, help="mesh resolution Nx=Ny")
parser.add_argument("--Nt", type=int, default=400, help="number of time steps")
parser.add_argument("--dt", type=float, default=1e-2, help="time step size")
parser.add_argument("--eps", type=float, default=5e-2, help="phase-field interface width")
parser.add_argument("--Dc", type=float, default=1e-4, help="VEGF diffusion coefficient")
parser.add_argument("--mu_c", type=float, default=1e-3, help="VEGF degradation rate")
parser.add_argument("--beta_c", type=float, default=1.0, help="coupling coefficient for phi equation")
parser.add_argument("--phi_pct", type=float, default=92.0, help="percentile threshold for initial phi")
parser.add_argument("--alpha_strength", type=float, default=1.0, help="VEGF source amplitude")
parser.add_argument("--alpha_radius", type=float, default=0.12, help="source radius (normalized)")
parser.add_argument("--alpha_cx", type=float, default=0.5, help="source center x (0~1)")
parser.add_argument("--alpha_cy", type=float, default=0.5, help="source center y (0~1)")
parser.add_argument("--mass_stab", action="store_true", help="enable phi mass stabilization")
parser.add_argument("--mobility_gate", action="store_true", help="enable mobility gating M(c)=m0+m1*min(c,cap)")
parser.add_argument("--m0", type=float, default=5e-3, help="baseline mobility (mobility_gate only)")
parser.add_argument("--m1", type=float, default=5e-2, help="mobility gain (mobility_gate only)")
parser.add_argument("--mcap", type=float, default=1.0, help="c saturation cap (mobility_gate only)")
parser.add_argument("--save_every", type=int, default=50, help="save interval (steps)")
parser.add_argument("--mass_k", type=float, default=0.3, help="mass correction strength [0,1]")
parser.add_argument("--auto_scale", action="store_true", help="auto-scale dt and eps by Nx")
parser.add_argument("--Nx_ref", type=int, default=128, help="reference Nx for auto_scale")
parser.add_argument("--eps_cells", type=float, default=5.0, help="interface width in cells")
parser.add_argument("--save_init", action="store_true", help="save t=0 fields")
parser.add_argument("--vessel_polarity", type=str, default="bright", choices=["bright", "dark"],
                    help="whether vessels are bright or dark in grayscale image")

args = parser.parse_args()
gray_image_folder = args.gray_folder or "gray_scales"

# --- output directories ---
os.makedirs("fig", exist_ok=True)
os.makedirs("res", exist_ok=True)

# --- mesh and function space ---
Nx = args.Nx
mesh = UnitSquareMesh(Nx, Nx)
V = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)
domain_area = assemble(Constant(1.0) * dx(domain=mesh))


# --- numpy-to-FEniCS interpolation ---
class NumpySampler(UserExpression):
    """Maps a 2D numpy array onto [0,1]^2 via nearest-neighbor sampling."""
    def __init__(self, img, **kwargs):
        super().__init__(**kwargs)
        self.img = np.array(img, dtype=float)
        self.ny, self.nx = self.img.shape
    def eval(self, values, xx):
        ix = int(round(xx[0]*(self.nx-1)))
        iy = int(round(xx[1]*(self.ny-1)))
        ix = max(0, min(ix, self.nx-1))
        iy = max(0, min(iy, self.ny-1))
        values[0] = float(self.img[iy, ix])
    def value_shape(self):
        return ()

def npy_to_Function(img_np, space):
    """Normalize numpy image to [0,1] and interpolate into FE space."""
    img = img_np.astype(float)
    rng = img.max() - img.min()
    if rng > 0:
        img = (img - img.min()) / rng
    return interpolate(NumpySampler(img=img, degree=1), space)


# --- field visualization ---
def save_field(u, path, title=""):
    import matplotlib.pyplot as plt
    plt.figure()
    c = plot(u)
    plt.colorbar(c)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# --- main loop: process each ROI file ---
raws = [f for f in sorted(os.listdir(gray_image_folder)) if f.endswith("_raw.npy")]
if not raws:
    print("No *_raw.npy files found in", gray_image_folder)

for gray_file in raws:
    print("Processing:", gray_file)
    A = np.load(os.path.join(gray_image_folder, gray_file))
    A = np.flipud(A)  # flip to match mesh y-axis orientation

    # initial phi: min-max normalize + threshold brightest pixels as vessels
    Amin, Amax = float(A.min()), float(A.max())
    rng = Amax - Amin
    An = (A - Amin) / rng if rng > 1e-12 else np.zeros_like(A)

    phi_pct = float(args.phi_pct)
    thr = np.percentile(An, phi_pct)
    phi0_np = (An >= thr).astype(float)

    # fallback: if coverage < 1%, lower threshold iteratively
    target_cover = 0.01
    min_pct = 30.0
    while float(phi0_np.mean()) < target_cover and phi_pct > min_pct:
        phi_pct -= 5.0
        thr = np.percentile(An, phi_pct)
        phi0_np = (An >= thr).astype(float)

    print(f"[init] bright-vessel: pct={phi_pct:.1f}, thr={thr:.3f}, cover={phi0_np.mean():.3e}")

    # light morphological denoising
    try:
        from scipy.ndimage import binary_opening, binary_closing
        phi0_np = binary_opening(phi0_np, structure=np.ones((3,3))).astype(float)
        phi0_np = binary_closing(phi0_np, structure=np.ones((3,3))).astype(float)
    except Exception:
        pass

    # save grayscale and overlay for visual check
    import matplotlib.pyplot as plt
    plt.figure(); plt.imshow(An, cmap="gray", vmin=0, vmax=1, origin="lower"); plt.title("An (min-max)"); plt.tight_layout()
    plt.savefig(f"fig/{gray_file[:-4]}_An.png", dpi=160); plt.close()

    plt.figure(); plt.imshow(An, cmap="gray", vmin=0, vmax=1, origin="lower")
    plt.contour(phi0_np, levels=[0.5], colors="r", linewidths=0.6)
    plt.title("Overlay: red = phi=1"); plt.tight_layout()
    plt.savefig(f"fig/{gray_file[:-4]}_overlay.png", dpi=160); plt.close()

    # alpha(x): VEGF source (treatment only)
    if args.group == "treatment":
        H, W = phi0_np.shape

        manual_center = (args.alpha_cx != 0.5) or (args.alpha_cy != 0.5)
        if manual_center:
            cx_use = float(args.alpha_cx)
            cy_use = float(args.alpha_cy)
        else:
            # auto-place source at strongest phi gradient (vessel boundary)
            g_y, g_x = np.gradient(phi0_np.astype(float))
            gmag = np.hypot(g_x, g_y)
            mask = (phi0_np > 0.5)
            if np.any(mask):
                iy, ix = np.unravel_index(np.argmax(gmag * mask), phi0_np.shape)
            else:
                iy, ix = np.unravel_index(np.argmax(gmag), phi0_np.shape)
            cx_use = float(ix / max(W - 1, 1))
            cy_use = float(iy / max(H - 1, 1))

        cx_use = min(max(cx_use, 0.0), 1.0)
        cy_use = min(max(cy_use, 0.0), 1.0)

        # 2D Gaussian source centered at (cx, cy)
        alpha = Expression(
            "a*exp(-((x[0]-cx)*(x[0]-cx)+(x[1]-cy)*(x[1]-cy))/(2*s*s))",
            degree=2, cx=cx_use, cy=cy_use, s=args.alpha_radius, a=args.alpha_strength
        )
    else:
        alpha = Constant(0.0)

    # initial c field: bump near source for treatment, zero for control
    if args.group == "treatment":
        cx_pix = int(round(cx_use * (A.shape[1]-1)))
        cy_pix = int(round(cy_use * (A.shape[0]-1)))
        R_pix = max(1, int(round(args.alpha_radius * max(A.shape))))
        Y, X = np.ogrid[:A.shape[0], :A.shape[1]]
        mask = (X-cx_pix)**2 + (Y-cy_pix)**2 <= R_pix*R_pix
        c0_np = np.zeros_like(An); c0_np[mask] = 1.0
    else:
        c0_np = np.zeros_like(An)

    # project initial fields to FE space
    phi = Function(V); phi.assign(npy_to_Function(phi0_np, V))
    c = Function(V); c.assign(npy_to_Function(c0_np, V))

    # check initial phi mass; if too small, lower threshold
    phi_mass0 = assemble(phi*dx)
    if float(phi_mass0) < 1e-4:
        thr2 = np.percentile(An, max(0.0, phi_pct - 2.0))
        phi0_np = (An >= thr2).astype(float)
        phi.assign(npy_to_Function(phi0_np, V))
        phi_mass0 = assemble(phi*dx)

    print("t=0 stats:",
          f"phi[min,max,mean]={phi.vector().min():.3e}, {phi.vector().max():.3e}, {assemble(phi*dx)/domain_area:.3e}")
    print("t=0 stats:",
          f"c[min,max,mean]={c.vector().min():.3e}, {c.vector().max():.3e}, {assemble(c*dx)/domain_area:.3e}")

    # store previous-step values for semi-implicit scheme
    phi_n = Function(V); phi_n.assign(phi)
    c_n = Function(V); c_n.assign(c)

    # trial/test functions and constants
    v = TestFunction(V)
    phi_trial = TrialFunction(V)
    c_trial = TrialFunction(V)

    Dc = Constant(args.Dc)
    mu = Constant(args.mu_c)
    beta = Constant(args.beta_c)

    h = 1.0 / float(Nx)

    # auto-scale dt and eps if requested
    if args.auto_scale:
        h_ref = 1.0 / float(args.Nx_ref)
        dt_auto = args.dt * (h / h_ref)**2
        eps_auto = max(args.eps, args.eps_cells * h)
        print(f"[auto_scale] h={h:.4e}, dt->{dt_auto:.4e}, eps->{eps_auto:.4e}")
        dt = Constant(dt_auto)
        eps = Constant(eps_auto)
    else:
        dt = Constant(args.dt)
        eps = Constant(args.eps)

    # mobility gating: M(c) = m0 + m1*min(c, cap)
    if args.mobility_gate:
        m0 = Constant(args.m0)
        m1 = Constant(args.m1)
        cap = Constant(args.mcap)
        M = project(m0 + m1*min_value(c_n, cap), V)
    else:
        M = Constant(0.0) if args.group == "control" else Constant(1.0)

    # variational forms (semi-implicit)
    # phi equation (Allen-Cahn):
    #   (phi^{n+1} - phi^n, v) + dt*M*eps^2*(grad(phi^{n+1}), grad(v))
    #   = -dt*M*8*phi^n*(1-phi^n)*(1-2*phi^n)*v  (double-well, explicit)
    #   + dt*beta*c^n*phi^n*(1-phi^n)*v           (VEGF coupling, explicit)
    Fphi_L = phi_n * v * dx \
           - dt * M * (8*phi_n*(1-phi_n)*(1-2*phi_n)) * v * dx \
           + dt * beta * c_n * (phi_n * (1-phi_n)) * v * dx
    Fphi_A = phi_trial * v * dx + dt * M * eps**2 * inner(grad(phi_trial), grad(v)) * dx
    a1 = lhs(Fphi_A); b1 = rhs(Fphi_L)

    # c equation (diffusion + degradation + source):
    #   (c^{n+1} - c^n, v) + dt*Dc_eff*(grad(c^{n+1}), grad(v)) + dt*mu*phi^n*c^{n+1}*v
    #   = dt*alpha*phi^n*v
    Dc_eff = Constant(1e-6) + Dc*phi_n  # small background diffusion for stability
    Fc_L = c_n*v*dx + dt * alpha * phi_n * v * dx
    Fc_A = c_trial*v*dx + dt * inner(Dc_eff*grad(c_trial), grad(v)) * dx + dt * mu * phi_n * c_trial * v * dx
    a2 = lhs(Fc_A); b2 = rhs(Fc_L)

    # time stepping
    phi_new = Function(V)
    c_new = Function(V)
    Nt = args.Nt

    if args.save_init:
        tag0 = f"{gray_file[:-4]}_t0000"
        save_field(phi, f"fig/{tag0}_phi.png", title=f"{tag0} phi (init)")
        save_field(c, f"fig/{tag0}_c.png", title=f"{tag0} c (init)")

    for n in range(1, Nt+1):
        # update mobility if gating enabled
        if args.mobility_gate:
            M.assign(project(m0 + m1*min_value(c_n, cap), V))

        # solve phi
        solve(a1 == b1, phi_new)

        # mass stabilization (optional)
        if args.mass_stab:
            if float(phi_mass0) > 1e-12:
                mass_now = assemble(phi_new*dx)
                corr = (mass_now - phi_mass0) / domain_area
                phi_new.vector()[:] -= float(args.mass_k) * corr

        # clamp phi to [0, 1]
        phi_new_np = phi_new.vector().get_local()
        phi_new_np = np.clip(phi_new_np, 0.0, 1.0)
        phi_new.vector().set_local(phi_new_np)
        phi_new.vector().apply("insert")

        if n <= 5 or n % 50 == 0:
            print(f"step {n}: phi[min,max,mean]="
                  f"{phi_new.vector().min():.3e}, "
                  f"{phi_new.vector().max():.3e}, "
                  f"{assemble(phi_new*dx)/domain_area:.3e}")

        # solve c
        solve(a2 == b2, c_new)

        # clamp c >= 0
        c_new_np = c_new.vector().get_local()
        c_new_np = np.maximum(c_new_np, 0.0)
        c_new.vector().set_local(c_new_np)
        c_new.vector().apply("insert")

        # advance
        phi_n.assign(phi_new)
        c_n.assign(c_new)

        # periodic save
        if n % args.save_every == 0 or n == Nt:
            tag = f"{gray_file[:-4]}_t{n:04d}"
            save_field(phi_new, f"fig/{tag}_phi.png", title=f"{tag} phi")
            save_field(c_new, f"fig/{tag}_c.png", title=f"{tag} c")
            np.savetxt(f"res/{tag}_phi.txt", phi_new.vector().get_local())
            np.savetxt(f"res/{tag}_c.txt", c_new.vector().get_local())

    print("Done:", gray_file)
