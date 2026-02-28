# Setup
# Cell 1: Setup
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plots
import data_loader
import importlib
importlib.reload(data_loader)


# print(f"Box size1: {BOX_SIZE}")

# Setup
from data_loader import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

print("BOX_SIZE2:", BOX_SIZE)


plt.rcParams.update({'font.size': 14})


df_corrected.head()


# Make Displacement Distribution Data

# Displacement Analysis for Multiple dt Values
dt_values = [1, 10, 50, 150]
Z_list = [] # Peak heights for each time lag

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()
plt.suptitle('Distribution of Bubble Displacements at Different Timesteps', 
             fontsize=20, y=0.95)

for idx, dt in enumerate(dt_values):
    all_dx, all_dy = [], []
    
    for bubble_id, g in df_corrected.groupby("id"):
        g = g.sort_values("timestep").copy()
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        
        if len(x) <= dt:
            continue
        
        dx_dt = x[dt:] - x[:-dt]
        dy_dt = y[dt:] - y[:-dt]
        
        all_dx.extend(dx_dt)
        all_dy.extend(dy_dt)
    
    all_disp = np.concatenate([all_dx, all_dy])
    counts, bins, patches = axs[idx].hist(all_disp, bins=80, alpha=0.9)
    axs[idx].set_title(rf"$\Delta t = {dt}$", fontsize=18)
    axs[idx].grid(alpha=0.3)
    axs[idx].set_yscale('log')
    axs[idx].set_xlim(-1.7, 1.7)
    axs[idx].set_ylim(1, 1e6)
    axs[idx].set_xlabel(r"Displacement $\Delta r$", fontsize=16)
    axs[idx].set_ylabel("Frequency")

    Z = 1/max(counts)
    Z_list.append(Z)
    
    print(f"Δt={dt}: n={len(all_disp)}, mean={np.mean(all_disp):.5f}, var={np.var(all_disp):.5f}")

plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.savefig("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/Figures_Sample2/03_Figures/displacement_distributions_2.png", dpi=300, bbox_inches='tight')
plt.show()

# print Z values
print("Z values for each dt:")
for dt, Z in zip(dt_values, Z_list):
    print(f"Δt={dt}: Z={Z:.5f}")


# Overlay with centred x and staggered (log) peaks
plt.figure(figsize=(8, 6))

markers = ['o', 's', '^', 'D', 'v', 'x']

# make sure order is from largest dt (highest) to smallest dt (lowest)
dt_plot_order = sorted(dt_values, reverse=True)

# choose where you want the top peak to sit and how much to separate peaks (in log10 decades)
top_peak = 2e4        # try 2e4 or 5e4 depending on your ylim
decade_step = 0.50    # ~0.50 decades separation (~3.16x in linear). Tweak to taste.

for idx, dt in enumerate(dt_plot_order):
    all_dx, all_dy = [], []

    for bubble_id, g in df_corrected.groupby("id"):
        g = g.sort_values("timestep").copy()
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        if len(x) <= dt:
            continue
        dx_dt = x[dt:] - x[:-dt]
        dy_dt = y[dt:] - y[:-dt]
        all_dx.extend(dx_dt)
        all_dy.extend(dy_dt)

    # concatenate and centre so the peak sits at x ~ 0
    all_disp = np.concatenate([all_dx, all_dy])
    all_disp -= all_disp.mean()

    # histogram as a probability density (PDF)
    bins = 80
    rng = (-1.7, 1.7)
    counts, bin_edges = np.histogram(all_disp, bins=bins, range=rng, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # scale so that peaks are staggered on a log axis
    current_peak = counts.max() if counts.max() > 0 else 1.0
    target_peak = top_peak * (10 ** (-decade_step * idx))  # 700 highest, then step down
    scale = target_peak / current_peak
    counts_scaled = counts * scale

    plt.plot(
        bin_centers, counts_scaled,
        marker=markers[idx % len(markers)], linestyle='none',
        label=rf'$\Delta t = {dt}$', alpha=0.85, markersize=4
    )

plt.yscale('log')
plt.xlim(-1.7, 1.7)
plt.ylim(0, 1e6)  # adjust if needed after you pick top_peak
plt.xlabel(r"Displacement", fontsize=16)
plt.ylabel("Probability density (staggered)", fontsize=16)
plt.title("Overlay: Bubble Displacements at Different Timesteps", fontsize=18)
plt.grid(alpha=0.3)
plt.legend(loc="upper right")
# reverse legend order
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles[::-1], labels[::-1], loc="upper right")
plt.tight_layout()
# plt.savefig("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/Figures_Sample2/03_Figures/displacement_overlay_shifted_peaks.png", dpi=300, bbox_inches='tight')
plt.show()

# reverse legend order
# change dt's


plt.figure(figsize=(8, 6))
markers = ['o', 's', '^', 'D', 'v', 'x']

# Choose the C (probability) levels for horizontal cuts
C_vals = [0.50, 0.15, 0.03]

# Plot largest dt first so it is on top in legend
dt_plot_order = sorted(dt_values, reverse=True)

for idx, dt in enumerate(dt_plot_order):
    all_dx, all_dy = [], []

    for bubble_id, g in df_corrected.groupby("id"):
        g = g.sort_values("timestep").copy()
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        if len(x) <= dt:
            continue
        dx_dt = x[dt:] - x[:-dt]
        dy_dt = y[dt:] - y[:-dt]
        all_dx.extend(dx_dt)
        all_dy.extend(dy_dt)

    # radial step sizes:
    # disp = np.sqrt(np.array(all_dx)**2 + np.array(all_dy)**2)
    disp = np.concatenate([all_dx, all_dy])

    vals = np.abs(disp) 
    vals_sorted = np.sort(vals)
    n = vals_sorted.size

    # A smooth X grid up to the 99.9th percentile
    X = np.linspace(0, np.quantile(vals_sorted, 0.999), 300)

    # Empirical CCDF: P(|x| > X)
    F = np.searchsorted(vals_sorted, X, side='right') / n 
    CCDF = 1.0 - F

    plt.plot(
        X, CCDF, marker=markers[idx % len(markers)],
        linestyle='none', ms=2, alpha=0.85,
        label=rf'$\Delta t = {dt}$'
    )

# Dashed lines for C values
for C in C_vals:
    plt.axhline(y=C, color='red', linestyle='--', linewidth=1)
    plt.text(
        1.02 * plt.xlim()[1], C, f" C = {C}",
        va='center', ha='left', fontsize=12, color='red'
    )

# Axis scaling and labels
plt.ylim(0, 1.02)
plt.xlim(left=0)
plt.ylabel(r"$P(|x|>X)$", fontsize=14)
plt.xlabel(r"$X$", fontsize=14)
plt.title("Complementary CDF of displacement components", fontsize=16)
plt.grid(alpha=0.3)
plt.legend(title=r"Lag $\Delta t$")
plt.tight_layout()
# plt.savefig("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/Figures_Sample2/03_Figures/ccdf_displacement_components_2.png", dpi=300, bbox_inches='tight')
plt.show()

# do it with simpsons and trapeziodal rule too

# === Horizontal cuts of CCDF and scaling plot ===

# choose the C levels
C_vals = [0.50, 0.15, 0.03]   

radial = False  # True = use r = sqrt(dx^2+dy^2), False = use |component| (Δx & Δy pooled)

def get_abs_displacements(dt):
    """Return |x| sample used in CCDF: either radial or component magnitude."""
    all_dx, all_dy = [], []
    for bubble_id, g in df_corrected.groupby("id"):
        g = g.sort_values("timestep").copy()
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        if len(x) <= dt:
            continue
        dx_dt = x[dt:] - x[:-dt]
        dy_dt = y[dt:] - y[:-dt]
        all_dx.extend(dx_dt)
        all_dy.extend(dy_dt)
    all_dx = np.asarray(all_dx)
    all_dy = np.asarray(all_dy)
    if radial:
        vals = np.sqrt(all_dx**2 + all_dy**2)
    else:
        vals = np.abs(np.concatenate([all_dx, all_dy]))
    return np.abs(vals)

# ensure dt=1 exists as the normalization reference
dt_plot_order = sorted(dt_values)
#assert dt_plot_order[0] == 1 or 1 in dt_plot_order, "Need Δt=1 in dt_values to form the ratio."

# precompute |x| samples for each Δt once (can be large)
abs_samples = {dt: get_abs_displacements(dt) for dt in dt_plot_order if len(get_abs_displacements(dt))}

# compute X(C, dt) for each level and dt
X_C_dt = {C: {} for C in C_vals}
for C in C_vals:
    q = 1.0 - C  # CCDF level C corresponds to quantile q=1-C
    for dt, vals in abs_samples.items():
        if len(vals) == 0:
            continue
        X_C_dt[C][dt] = float(np.quantile(vals, q))

# make the scaling plot: X(C, dt) / X(C, 1) vs dt
plt.figure(figsize=(8, 6))
markers = ['o', 's', '^', 'D', 'v', 'x']

for i, C in enumerate(C_vals):
    dts = np.array(sorted(X_C_dt[C].keys()))
    Xs  = np.array([X_C_dt[C][dt] for dt in dts])
    # normalize by dt=1 value (guard if missing)
    if 1 not in X_C_dt[C]:
        raise ValueError("dt=1 missing; cannot normalize.")
    X1 = X_C_dt[C][1]
    ratio = Xs / X1

    # log–log fit 
    # ratio ≈ A * dt^beta  => log(ratio) = log(A) + beta*log(dt)
    mask = dts != 1         # exclude dt==1;
    dts_fit = dts[mask]
    ratio_fit = ratio[mask]

    beta, logA = np.polyfit(np.log(dts_fit), np.log(ratio_fit), 1)[0], np.polyfit(np.log(dts_fit), np.log(ratio_fit), 1)[1]
    A = np.exp(logA)
    fit_curve = A * dts**beta

    plt.loglog(dts, ratio, marker=markers[i % len(markers)], linestyle='none',
               label=fr'$C={C}$  (slope $\beta\approx{beta:.2f}$)')
    plt.loglog(dts, fit_curve, linestyle='-', alpha=0.6)

plt.xlabel(r'$\Delta t$', fontsize=14)
plt.ylabel(r'$X(C,\Delta t)/X(C,1)$', fontsize=14)
plt.title('Scaling of CCDF cuts: $X(C,\Delta t)$ relative to $\Delta t=1$', fontsize=16)
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
# plt.savefig("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/Figures_Sample2/03_Figures/ccdf_cut_scaling_2.png", dpi=300, bbox_inches='tight')
plt.show()

# try plotting log of data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


# ------------------------------------------------------------------ #
#  model
# ------------------------------------------------------------------ #
def p_model(x, beta, alpha, Z):
    return (1.0 / Z) * (1.0 + (beta * x**2) / alpha) ** (-alpha)


# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #
def pooled_displacements(df_corrected, dt, use_S=False):
    all_d = []
    for _, g in df_corrected.groupby("id"):
        g = g.sort_values("timestep")
        X = g["Sx" if use_S else "x"].to_numpy()
        Y = g["Sy" if use_S else "y"].to_numpy()
        if len(X) <= dt:
            continue
        all_d += [X[dt:] - X[:-dt], Y[dt:] - Y[:-dt]]
    return np.concatenate(all_d) if all_d else np.array([])


def make_histogram(disp, nbins=160, x_cut_q=0.995):
    x_max = np.quantile(np.abs(disp), x_cut_q)
    bins  = np.linspace(-x_max, x_max, nbins + 1)
    p, edges = np.histogram(disp, bins=bins, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, p


def peak_Z(x, p):
    """Z estimated from the peak bin (bin nearest 0)."""
    i0 = np.argmin(np.abs(x))
    P0 = p[i0]
    if P0 <= 0:
        raise ValueError("Peak bin has zero/negative density — try more bins.")
    return 1.0 / P0


def beta_from_variance(disp, alpha):
    return alpha / (np.var(disp) + 1e-30)


# ================================================================== #
#  ===RUN===  — edit these three lines, everything else is automatic
# ================================================================== #
# from your_notebook import df_corrected   # ← uncomment / adjust
alpha_fixed  = 1.54          # fixed alpha
dt_values    = [1, 10, 50, 150]
use_S        = False         # True → use Sx/Sy columns
nbins        = 160
# ================================================================== #


# ------------------------------------------------------------------ #
#  precompute histograms
# ------------------------------------------------------------------ #
hists = {}
for dt in dt_values:
    disp = pooled_displacements(df_corrected, dt, use_S=use_S)
    if disp.size < 50:
        print(f"dt={dt}: too few points, skipping")
        continue
    x, p = make_histogram(disp, nbins=nbins)
    Z0   = peak_Z(x, p)
    b0   = beta_from_variance(disp, alpha_fixed)
    hists[dt] = dict(x=x, p=p, Z0=Z0, beta0=b0, Z_cur=Z0, beta_cur=b0)

dt_list = sorted(hists.keys())
if not dt_list:
    raise RuntimeError("No dt values produced histograms.")

cur_dt = dt_list[0]


# ------------------------------------------------------------------ #
#  build figure
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("#1a1a2e")

# main axes
ax = fig.add_axes([0.08, 0.30, 0.88, 0.62])
ax.set_facecolor("#16213e")
for sp in ax.spines.values():
    sp.set_color("#e0e0e0")
ax.tick_params(colors="#e0e0e0")
ax.xaxis.label.set_color("#e0e0e0")
ax.yaxis.label.set_color("#e0e0e0")
ax.title.set_color("#e0e0e0")

def _plot(dt):
    h = hists[dt]
    x, p = h["x"], h["p"]
    mask = p > 0
    ax.clear()
    ax.set_facecolor("#16213e")
    ax.scatter(x[mask], p[mask], s=6, color="#64b5f6", alpha=0.7, label="data")

    x_line = np.linspace(x.min(), x.max(), 2000)
    y_line = p_model(x_line, h["beta_cur"], alpha_fixed, h["Z_cur"])
    ax.plot(x_line, y_line, lw=2.2, color="#ff6b6b",
            label=rf"model  $\beta$={h['beta_cur']:.4g}  Z={h['Z_cur']:.4g}  $\alpha$={alpha_fixed}")

    # mark peak-Z level
    ax.axhline(1.0 / h["Z_cur"], color="#ffd54f", lw=0.8, ls="--", alpha=0.6, label="1/Z level")

    ax.set_yscale("log")
    ax.set_xlabel("displacement", color="#e0e0e0")
    ax.set_ylabel("p(x, Δt)", color="#e0e0e0")
    ax.set_title(rf"$\Delta t = {dt}$   [α fixed = {alpha_fixed}]", color="#e0e0e0")
    ax.legend(facecolor="#0f3460", labelcolor="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#e0e0e0")
    for sp in ax.spines.values():
        sp.set_color("#e0e0e0")
    fig.canvas.draw_idle()

_plot(cur_dt)


# ------------------------------------------------------------------ #
#  sliders
# ------------------------------------------------------------------ #
sl_kw = dict(facecolor="#0f3460", color="#64b5f6")

ax_beta = fig.add_axes([0.15, 0.18, 0.70, 0.03], facecolor="#0f3460")
ax_Z    = fig.add_axes([0.15, 0.12, 0.70, 0.03], facecolor="#0f3460")

h0 = hists[cur_dt]
s_beta = Slider(ax_beta, r"$\beta$",  0.001, h0["beta0"]*6,
                valinit=h0["beta_cur"], color="#ff6b6b")
s_Z    = Slider(ax_Z,    "Z",         0.01,  h0["Z0"]*6,
                valinit=h0["Z_cur"],   color="#ffd54f")

# style slider labels
for sl in (s_beta, s_Z):
    sl.label.set_color("#e0e0e0")
    sl.valtext.set_color("#e0e0e0")

z_locked = [True]   # mutable flag

def _sync_sliders(dt):
    h = hists[dt]
    # rescale ranges sensibly
    s_beta.valmin = 0.001
    s_beta.valmax = h["beta0"] * 8
    s_beta.ax.set_xlim(s_beta.valmin, s_beta.valmax)
    s_beta.set_val(h["beta_cur"])

    s_Z.valmin = h["Z0"] * 0.1
    s_Z.valmax = h["Z0"] * 8
    s_Z.ax.set_xlim(s_Z.valmin, s_Z.valmax)
    s_Z.set_val(h["Z_cur"])
    s_Z.active = not z_locked[0]

def on_beta(val):
    hists[cur_dt]["beta_cur"] = val
    _plot(cur_dt)

def on_Z(val):
    if not z_locked[0]:
        hists[cur_dt]["Z_cur"] = val
        _plot(cur_dt)

s_beta.on_changed(on_beta)
s_Z.on_changed(on_Z)


# ------------------------------------------------------------------ #
#  dt radio buttons
# ------------------------------------------------------------------ #
ax_radio = fig.add_axes([0.01, 0.55, 0.06, 0.25], facecolor="#0f3460")
radio = RadioButtons(ax_radio, [str(d) for d in dt_list],
                     activecolor="#ff6b6b")
for lbl in radio.labels:
    lbl.set_color("#e0e0e0")
    lbl.set_fontsize(9)

def on_radio(label):
    global cur_dt
    cur_dt = int(label)
    _sync_sliders(cur_dt)
    _plot(cur_dt)

radio.on_clicked(on_radio)


# ------------------------------------------------------------------ #
#  buttons
# ------------------------------------------------------------------ #
btn_kw = dict(color="#0f3460", hovercolor="#1a4a7a")

ax_unlock = fig.add_axes([0.15, 0.04, 0.18, 0.05])
btn_unlock = Button(ax_unlock, "🔓 Unlock Z", **btn_kw)
btn_unlock.label.set_color("#ffd54f")

ax_lock = fig.add_axes([0.35, 0.04, 0.18, 0.05])
btn_lock = Button(ax_lock, "🔒 Lock Z to peak", **btn_kw)
btn_lock.label.set_color("#64b5f6")

ax_reset = fig.add_axes([0.55, 0.04, 0.18, 0.05])
btn_reset = Button(ax_reset, "↺ Reset", **btn_kw)
btn_reset.label.set_color("#e0e0e0")

ax_print = fig.add_axes([0.75, 0.04, 0.18, 0.05])
btn_print = Button(ax_print, "⎙ Print params", **btn_kw)
btn_print.label.set_color("#aaffaa")

def on_unlock(event):
    z_locked[0] = False
    s_Z.active = True
    print(f"[dt={cur_dt}] Z unlocked — current Z={hists[cur_dt]['Z_cur']:.6g}")

def on_lock(event):
    z_locked[0] = True
    s_Z.active = False
    hists[cur_dt]["Z_cur"] = hists[cur_dt]["Z0"]
    s_Z.set_val(hists[cur_dt]["Z0"])
    _plot(cur_dt)
    print(f"[dt={cur_dt}] Z re-locked to peak: Z={hists[cur_dt]['Z0']:.6g}")

def on_reset(event):
    h = hists[cur_dt]
    h["beta_cur"] = h["beta0"]
    h["Z_cur"]    = h["Z0"]
    z_locked[0]   = True
    _sync_sliders(cur_dt)
    _plot(cur_dt)

def on_print(event):
    print("\n--- Current fit parameters ---")
    for dt in dt_list:
        h = hists[dt]
        print(f"  dt={dt:>4}:  beta={h['beta_cur']:.6g}   Z={h['Z_cur']:.6g}   "
              f"(Z0_peak={h['Z0']:.6g})")
    print()

btn_unlock.on_clicked(on_unlock)
btn_lock.on_clicked(on_lock)
btn_reset.on_clicked(on_reset)
btn_print.on_clicked(on_print)

# ------------------------------------------------------------------ #
fig.suptitle("Interactive p(x) fitter  —  tune β first, then unlock Z",
             color="#e0e0e0", fontsize=13, y=0.99)
plt.show()


print(matplotlib.get_backend())  # "agg" = broken, "Qt5Agg" / "TkAgg" = fine