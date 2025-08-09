"""
transport_analysis.py
Synthetic transport data -> analysis -> PDF report

Dependencies: numpy, pandas, matplotlib, scipy
Produces: outputs/plots/*.png and outputs/report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import poisson

# -----------------------
# Configuration / seed
# -----------------------
RANDOM_SEED = 2025  # change for a different dataset
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------
# Problem setup (corridor with 3 links + 1 intersection)
# -----------------------
hours = np.arange(0, 24)
n_links = 3
links = [f"Link_{i+1}" for i in range(n_links)]
# nominal capacities (veh/hr), free-flow speeds (kph), jam density (veh/km)
link_params = {
    "Link_1": {"capacity": 1800, "u_f": 60, "k_j": 150, "length_km": 1.2},
    "Link_2": {"capacity": 1500, "u_f": 50, "k_j": 140, "length_km": 0.8},
    "Link_3": {"capacity": 2000, "u_f": 70, "k_j": 160, "length_km": 1.5},
}

# base hourly demand profile (proportion of daily demand)
base_profile = np.array([
    0.02,0.015,0.01,0.01,0.02,0.03, # 0-5
    0.06,0.08,0.10,0.09,0.07,0.05,   # 6-11
    0.04,0.04,0.05,0.07,0.08,0.09,   # 12-17
    0.06,0.04,0.03,0.02,0.02,0.02    # 18-23
])
base_profile = base_profile / base_profile.sum()

# total daily demand per link (veh/day) — randomize a little per link
daily_demands = {
    link: int(params["capacity"] * np.random.uniform(1.2, 1.6, size=1)[0])  # generate daily demand proportional to capacity
    for link, params in link_params.items()
}

# -----------------------
# Generate synthetic hourly counts (Poisson around expected)
# -----------------------
records = []
for link in links:
    daily = daily_demands[link]
    expected_hourly = daily * base_profile
    # Poisson random sample per hour
    hourly_counts = poisson.rvs(expected_hourly, random_state=None)
    for hr, cnt in zip(hours, hourly_counts):
        records.append({"link": link, "hour": int(hr), "count": int(cnt)})

df_counts = pd.DataFrame(records)

# -----------------------
# Analysis functions
# -----------------------
def greenshields_speed(u_f, k, k_j):
    """Greenshields linear speed-density: u = u_f * (1 - k/k_j)"""
    return np.maximum(0.0, u_f * (1 - k / k_j))

def mm1_delay(lambda_rate, mu_rate):
    """Mean waiting time in M/M/1 (hours). If rho>=1 return np.inf"""
    if lambda_rate >= mu_rate:
        return np.inf
    rho = lambda_rate / mu_rate
    Wq = rho / (mu_rate - lambda_rate)  # hours
    return Wq

# -----------------------
# Link-level metrics per hour
# -----------------------
analysis_rows = []
for link in links:
    params = link_params[link]
    cap = params["capacity"]
    u_f = params["u_f"]
    k_j = params["k_j"]
    L = params["length_km"]
    for hr in hours:
        q = int(df_counts[(df_counts.link==link) & (df_counts.hour==hr)]["count"].values[0])  # veh/hr
        # estimate density using q = k * u -> k = q / u_est.
        # Use Greenshields: u = u_f (1 - k/k_j) => q = k * u_f (1 - k/k_j) -> quadratic in k.
        # Solve for k numerically: k * u_f (1 - k/k_j) - q = 0
        # rearrange: - (u_f / k_j) * k^2 + u_f * k - q = 0
        a = -(u_f / k_j)
        b = u_f
        c = -q
        # quadratic solution
        disc = b*b - 4*a*c
        if disc < 0:
            k = max(0.0, q / u_f)  # fallback
        else:
            k1 = (-b + np.sqrt(disc)) / (2*a)
            k2 = (-b - np.sqrt(disc)) / (2*a)
            # choose realistic positive root less than jam density
            k = max(min(k1, k_j), 0.0)
            if not (0 <= k <= k_j):
                k = max(min(k2, k_j), 0.0)
        u = greenshields_speed(u_f, k, k_j)  # kph
        v_by_c = q / cap if cap>0 else np.nan
        # Treat as M/M/1 queue with service rate mu = capacity (veh/hr)
        lam = q
        mu = cap
        Wq = mm1_delay(lam, mu)  # hours
        # Convert times to minutes for readability
        Wq_min = np.inf if np.isinf(Wq) else Wq * 60.0
        travel_time_min = (L / u * 60.0) if u>0 else np.inf
        analysis_rows.append({
            "link": link, "hour": hr, "flow_vph": q, "density_veh_per_km": k,
            "speed_kph": u, "v_by_c": v_by_c, "queue_delay_min": Wq_min,
            "travel_time_min": travel_time_min
        })

df_analysis = pd.DataFrame(analysis_rows)

# -----------------------
# Summaries
# -----------------------
peak_info = df_analysis.groupby("link").apply(lambda g: g.loc[g.flow_vph.idxmax(), ["hour","flow_vph"]]).reset_index()
peak_info.columns = ["link","peak_hour","peak_flow_vph"]

# Level of Service (very simplified using v/c and travel time):
def los_from_vc(v_by_c):
    if v_by_c < 0.6: return "A"
    if v_by_c < 0.7: return "B"
    if v_by_c < 0.8: return "C"
    if v_by_c < 0.9: return "D"
    if v_by_c < 1.0: return "E"
    return "F"

df_analysis["LOS"] = df_analysis["v_by_c"].apply(los_from_vc)

# -----------------------
# Save tables
# -----------------------
df_counts.to_csv(os.path.join(OUTPUT_DIR, "hourly_counts.csv"), index=False)
df_analysis.to_csv(os.path.join(OUTPUT_DIR, "link_hourly_analysis.csv"), index=False)
peak_info.to_csv(os.path.join(OUTPUT_DIR, "peak_summary.csv"), index=False)

# -----------------------
# Plots
# -----------------------
for link in links:
    g = df_analysis[df_analysis.link==link]
    plt.figure(figsize=(8,4))
    plt.plot(g.hour, g.flow_vph, marker='o')
    plt.title(f"{link} - Hourly Flow (vph)")
    plt.xlabel("Hour")
    plt.ylabel("Flow (veh/h)")
    plt.grid(True)
    pfile = os.path.join(PLOTS_DIR, f"{link}_flow.png")
    plt.savefig(pfile, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(g.hour, g.speed_kph, marker='o')
    plt.title(f"{link} - Estimated Speed (kph) via Greenshields")
    plt.xlabel("Hour")
    plt.ylabel("Speed (kph)")
    plt.grid(True)
    pfile = os.path.join(PLOTS_DIR, f"{link}_speed.png")
    plt.savefig(pfile, bbox_inches="tight")
    plt.close()

# Combined v/c heatmap-like plot (simpler: scatter)
plt.figure(figsize=(8,4))
for link in links:
    g = df_analysis[df_analysis.link==link]
    plt.scatter(g.hour + 0.05*links.index(link), g.v_by_c, label=link)
plt.axhline(1.0, color='gray', linestyle='--')
plt.title("Hour vs v/c (by link)")
plt.xlabel("Hour")
plt.ylabel("v/c")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "vc_scatter.png"), bbox_inches="tight")
plt.close()

# -----------------------
# Create PDF report
# -----------------------
report_path = os.path.join(OUTPUT_DIR, "report.pdf")
with PdfPages(report_path) as pdf:
    # Page 1: Title + summary text
    fig = plt.figure(figsize=(11,8.5))
    fig.text(0.1, 0.85, "Transport Analysis Report (Synthetic Data)", fontsize=18, weight='bold')
    fig.text(0.1, 0.80, f"Generated with RANDOM_SEED={RANDOM_SEED}", fontsize=10)
    fig.text(0.1, 0.74, "Study description:", fontsize=12, weight='bold')
    desc = ("Corridor with 3 links. Synthetic hourly counts generated via Poisson sampling. "
            "Speeds estimated with Greenshields model. Queue delays estimated via M/M/1 approximation. "
            "This report contains time-series plots and tables.")
    fig.text(0.1, 0.66, desc, fontsize=10, wrap=True)
    # Add peaks
    y = 0.60
    for _, row in peak_info.iterrows():
        fig.text(0.1, y, f"{row['link']}: peak hour = {int(row['peak_hour'])}, peak flow = {int(row['peak_flow_vph'])} veh/h", fontsize=10)
        y -= 0.03
    pdf.savefig(fig)
    plt.close()

    # Page 2..n: include plots
    for img in sorted(os.listdir(PLOTS_DIR)):
        fig = plt.figure(figsize=(11,8.5))
        ax = fig.add_subplot(111)
        img_path = os.path.join(PLOTS_DIR, img)
        im = plt.imread(img_path)
        ax.imshow(im)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close()

    # Page last: include a small table snapshot (first few rows)
    fig = plt.figure(figsize=(11,8.5))
    fig.text(0.05, 0.95, "Sample of Analysis Table (first 10 rows)", fontsize=12, weight='bold')
    tbl = df_analysis.head(10).round(2)
    table_text = tbl.to_string(index=False)
    fig.text(0.05, 0.05, table_text, fontsize=8, family='monospace')
    pdf.savefig(fig)
    plt.close()

print(f"Report written to {report_path}")
print(f"CSV outputs written to {OUTPUT_DIR}")
