"""
Bubble Data Loader Module
Loads and preprocesses bubble foam simulation data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# DATA_FILE = Path("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/DataFiles/wetfoam2_bub_RH2_0.020000_0.507713.txt")
DATA_FILE = Path("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/DataFiles/wetfoam_bub_RH2_0.080000_0.507713.txt")
# DATA_FILE = Path("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/DataFiles/wetfoam_bub_RH2_0.140000_0.502800.txt")
# DATA_FILE = Path("/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/DataFiles/wetfoam3_bub_RH3_0.080000_0.501221.txt")

# BOX_SIZE = 20.203051
BOX_SIZE = 20.851441
# BOX_SIZE = 20.0
# BOX_SIZE = 39.562828

half_box = BOX_SIZE / 2.0
PERIODIC_THRESHOLD = 10.0
LIQUID_FRACTION = 0.02

# ============================================================================
# LOAD RAW DATA
# ============================================================================

print("Loading bubble data...")
df = pd.read_csv(
    DATA_FILE,
    comment="#",
    sep=r"\s+",
    header=None,
    names=["id", "x", "y", "area", "pressure", "Z"]
)
print(f"✓ Loaded {len(df)} observations")

# ============================================================================
# ADD TIMESTEPS
# ============================================================================

timesteps = []
step = -1
with DATA_FILE.open() as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.lower().startswith("#id"):
                step += 1
            continue
        timesteps.append(step)

df["timestep"] = timesteps
final_step = df["timestep"].max()
print(f"✓ Timesteps added: 0 to {final_step}")

# ============================================================================
# IDENTIFY DISAPPEARING BUBBLES
# ============================================================================

disappearing_ids = set(
    df.groupby("id")["timestep"].max()
      .loc[lambda s: s < final_step]
      .index
)
print(f"✓ Identified {len(disappearing_ids)} disappearing bubbles")

# ============================================================================
# APPLY PERIODIC BOUNDARY CORRECTION
# ============================================================================

print("Applying periodic boundary corrections...")

# Ensure a clean, unique, simple index
df_corrected = df.reset_index(drop=True).copy()

# Work on raw NumPy arrays indexed by *row number*
x_all = df_corrected["x"].to_numpy(dtype=float)
y_all = df_corrected["y"].to_numpy(dtype=float)

for bubble_id, g in df_corrected.groupby("id"):
    # These are integer row positions into x_all / y_all
    idx = g.index.to_numpy()

    # Extract this bubble's trajectory
    x = x_all[idx].copy()
    y = y_all[idx].copy()

    # Sort by timestep *within the group* (important!)
    order = np.argsort(g["timestep"].to_numpy())
    idx_sorted = idx[order]
    x = x[order]
    y = y[order]

    # Apply periodic corrections along the sorted track
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]

        if dx > half_box:
            x[i:] -= BOX_SIZE
        elif dx < -half_box:
            x[i:] += BOX_SIZE

        if dy > half_box:
            y[i:] -= BOX_SIZE
        elif dy < -half_box:
            y[i:] += BOX_SIZE

    # Write the corrected values back into the global arrays
    x_all[idx_sorted] = x
    y_all[idx_sorted] = y

# After all groups processed, assign back to dataframe *once*
df_corrected["x"] = x_all
df_corrected["y"] = y_all

print("✓ Periodic boundary correction applied")

# ============================================================================
# COMPUTE DERIVED QUANTITIES
# ============================================================================

# Add actual area column (area column contains radius)
df_corrected['actual_area'] = np.pi * df_corrected['area']**2

# Bubbles per timestep
bubbles_per_timestep = df_corrected.groupby("timestep")["id"].nunique()

# Average area per timestep
avg_area_per_timestep = df_corrected.groupby("timestep")["area"].apply(
    lambda x: np.mean(np.pi * x**2)
)
A_0 = avg_area_per_timestep.iloc[0]  # Initial average area

# Box area
box_area = BOX_SIZE ** 2

# Approximated area using liquid fraction
approx_avg_area = (box_area * (1 - LIQUID_FRACTION)) / bubbles_per_timestep

# Max timestep
max_timestep = df_corrected["timestep"].max()

print("✓ Derived quantities computed")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("DATA LOADING COMPLETE")
print("="*60)
print(f"Total observations:       {len(df_corrected):,}")
print(f"Unique bubbles:           {df_corrected['id'].nunique()}")
print(f"Timesteps:                0 to {final_step}")
print(f"Disappearing bubbles:     {len(disappearing_ids)}")
print(f"Box size:                 {BOX_SIZE}")
print(f"Initial avg area (A_0):   {A_0:.6f}")
print("="*60)
print("\nAvailable variables:")
print("  - df: Original dataframe")
print("  - df_corrected: Corrected dataframe with actual_area column")
print("  - disappearing_ids: Set of disappearing bubble IDs")
print("  - bubbles_per_timestep: Number of bubbles at each timestep")
print("  - avg_area_per_timestep: Average bubble area at each timestep")
print("  - approx_avg_area: Approximate area from liquid fraction")
print("  - A_0: Initial average bubble area")
print("  - max_timestep: Maximum timestep")
print("  - final_step: Final timestep")
print("  - box_area: Total box area")
print("  - BOX_SIZE, PERIODIC_THRESHOLD, LIQUID_FRACTION: Constants")
print("="*60 + "\n")