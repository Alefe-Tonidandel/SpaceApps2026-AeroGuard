import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent / 'src'))
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import firms_api
import harmonize as H
import history_tab
import product_view

st.set_page_config(page_title="AeroGuard | MODIS & VIIRS Harmonizer", layout="wide")

st.title("AeroGuard: MODIS & VIIRS Hotspot Harmonization Calendar")
st.markdown(
    "**NASA Space Apps Challenge** — Harmonizing **MODIS (1 km)** and **VIIRS (375 m)** "
    "active fire detections via **Real Pixel Footprint Normalization ($\\text{scan} \\times \\text{track}$)**, "
    "**2D Spatial Grid Quantization**, and **Separable Digital Image Processing Filters**."
)


def load_local_key() -> str:
    k = os.environ.get("FIRMS_KEY", "").strip()
    if not k:
        try:
            k = str(st.secrets.get("FIRMS_KEY", "")).strip()
        except Exception:
            pass
    if not k and os.path.exists(".streamlit/secrets.toml"):
        with open(".streamlit/secrets.toml", "r", encoding="utf-8") as f:
            for line in f:
                if "FIRMS_KEY" in line and "=" in line:
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k


default_key = load_local_key()

st.sidebar.header("Harmonization Parameters")
user_key = st.sidebar.text_input(
    "NASA FIRMS MAP_KEY (Optional if in secrets)",
    value="",
    type="password",
    help="Leave blank to use .streamlit/secrets.toml or cached local CSV datasets."
)
active_key = user_key.strip() if user_key.strip() else default_key

regioes = {
    "Minas Gerais - Sep/Oct Peak Season [48.6k rows | 4 Sats] (-51,-23,-39,-14)": ("-51,-23,-39,-14", "data/dados_incendios_mg_set.csv"),
    "Minas Gerais - Nov Recent 10d [2.7k rows | 3 Sats] (-51,-23,-39,-14)": ("-51,-23,-39,-14", "data/dados_incendios_mg.csv"),
    "Brazil Macro Bounding Box (-74,-34,-34,5)": ("-74,-34,-34,5", "data/dados_incendios.csv"),
}
regiao_label = st.sidebar.selectbox("Dataset / Area of Interest", list(regioes.keys()))
bbox, target_csv = regioes[regiao_label]

dias = st.sidebar.slider(
    "API Time Window (Days)",
    min_value=1,
    max_value=10,
    value=10,
    help="Fetched in 5-day windows via firms_api.py."
)
grid_res = st.sidebar.select_slider(
    "Harmonization Grid Resolution (Degrees)",
    options=[0.01, 0.02, 0.05, 0.1],
    value=0.02,
    help="0.02° (~2.2 km) is recommended so off-nadir MODIS pixels rarely exceed the cell area."
)

conf_options = st.sidebar.multiselect(
    "Unified NASA Confidence Classes",
    options=["low", "nominal", "high"],
    default=["nominal", "high"],
    help="Standardized across MODIS (0-29 low, 30-79 nominal, 80-100 high) and VIIRS (l, n, h)."
)
remove_qa_suspect = st.sidebar.checkbox("Filter Physically Incoherent Pixels (T < 290 K)", value=True)
use_brt_tz = st.sidebar.checkbox("Use Local Brazil Time (BRT / UTC-3) for Calendar", value=True)
exclude_partial_edges = st.sidebar.checkbox("Exclude Incomplete Edge Days in Threshold Calc", value=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_firms_live(key: str, area: str, d: int):
    return firms_api.fetch_all(key, area, d)


@st.cache_data(show_spinner=False)
def run_pipeline(df_in: pd.DataFrame, bbox_str: str, res: float, allowed_conf: tuple, filter_qa: bool, use_local_tz: bool):
    w, s, e, n = map(float, bbox_str.split(","))
    df_box = df_in.copy()
    df_box["latitude"] = pd.to_numeric(df_box["latitude"], errors="coerce")
    df_box["longitude"] = pd.to_numeric(df_box["longitude"], errors="coerce")
    df_box = df_box[df_box["longitude"].between(w, e) & df_box["latitude"].between(s, n)]

    if df_box.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    d, qa = H.validate_and_prepare(df_box)

    if "acq_time" in d.columns:
        dt_utc = pd.to_datetime(
            d["acq_date"].astype(str) + " " + d["acq_time"].astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M",
            errors="coerce",
        )
    elif "data_hora" in d.columns:
        dt_utc = pd.to_datetime(d["data_hora"], errors="coerce")
    else:
        dt_utc = pd.to_datetime(d["acq_date"], errors="coerce")

    d["data_hora_utc"] = dt_utc
    if use_local_tz and dt_utc.notna().any():
        dt_brt = dt_utc - pd.Timedelta(hours=3)
        d["acq_date"] = dt_brt.dt.strftime("%Y-%m-%d").fillna(d["acq_date"])
        d["hour_plot"] = dt_brt.dt.hour.fillna(0).astype(int)
    else:
        d["hour_plot"] = dt_utc.dt.hour.fillna(0).astype(int) if dt_utc.notna().any() else 0

    mask = d["conf_class"].isin(allowed_conf)
    if filter_qa:
        mask = mask & (~d["qa_suspect"])
    d_filt = d[mask].copy()

    if d_filt.empty:
        return d_filt, pd.DataFrame(), pd.DataFrame(), qa

    cells, per = H.harmonize(d_filt, grid_res=res)
    return d_filt, cells, per, qa


st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
if "df_raw" not in st.session_state or st.session_state.get("loaded_target") != target_csv:
    for candidate in [target_csv, "data/dados_incendios_mg.csv", "data/dados_incendios_mg_set.csv", "data/dados_incendios.csv"]:
        if os.path.exists(candidate):
            st.session_state["df_raw"] = pd.read_csv(candidate)
            st.session_state["loaded_target"] = target_csv
            st.session_state["logs"] = [f"Loaded local file: {candidate} ({len(st.session_state['df_raw'])} rows)"]
            break
    else:
        st.session_state["df_raw"] = pd.DataFrame()
        st.session_state["logs"] = []

c_btn1, c_btn2 = st.sidebar.columns(2)
if c_btn1.button("Fetch Live API"):
    if not active_key:
        st.sidebar.error("No FIRMS_KEY found. Add it to .streamlit/secrets.toml or paste above.")
    else:
        with st.spinner(f"Fetching {dias} days in 5-day windows..."):
            df_api, logs, all_ok = fetch_firms_live(active_key, bbox, dias)
            st.session_state["logs"] = logs
            if not df_api.empty:
                st.session_state["df_raw"] = df_api
                out_name = "data/dados_incendios_live_api.csv"
                df_api.to_csv(out_name, index=False)
            else:
                st.sidebar.warning("API returned no data; keeping current dataset in memory.")

if c_btn2.button("Reload CSV"):
    for candidate in [target_csv, "data/dados_incendios_mg.csv", "data/dados_incendios_mg_set.csv", "data/dados_incendios.csv"]:
        if os.path.exists(candidate):
            st.session_state["df_raw"] = pd.read_csv(candidate)
            st.session_state["logs"] = [f"Loaded {candidate} ({len(st.session_state['df_raw'])} rows)"]
            break

for l in st.session_state.get("logs", []):
    st.sidebar.caption(l)

df_raw = st.session_state["df_raw"]
if df_raw.empty:
    st.warning("No dataset loaded. Run `python3 coleta.py` or click 'Fetch Live API'.")
    st.stop()

try:
    d_filt, cells, per, qa = run_pipeline(
        df_raw, bbox, grid_res, tuple(conf_options), remove_qa_suspect, use_brt_tz
    )
except Exception as e:
    st.error(f"Validation error: {e}")
    st.stop()

if d_filt.empty or cells.empty:
    st.warning("No valid hotspots inside the selected Bounding Box or filter criteria.")
    st.stop()

n_oversized = int(cells["pixel_gt_cell"].sum())
if n_oversized > 0:
    st.info(
        f"**Footprint Notice:** {n_oversized} of {len(cells)} harmonized cells ({n_oversized/len(cells)*100:.1f}%) "
        f"contain off-nadir pixels larger than the {grid_res}° cell area (~{cells['cell_km2'].median():.2f} km²). "
        f"Use ≥ 0.02° to minimize footprint spillover."
    )

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Validated Detections", f"{len(d_filt):,}", f"{qa.get('qa_suspect', 0)} QA suspect (T<290K)")
k2.metric("Harmonized Fire Cells", f"{len(cells):,}", f"Grid: {grid_res}° (~{cells['cell_km2'].median():.1f} km²)")
intra_reduction = (1 - len(cells) / max(len(d_filt), 1)) * 100
k3.metric("Spatial Clustering Ratio", f"{intra_reduction:.1f}%", "Intra-sensor & overpass aggregation")
n_cross = int((cells["n_families"] > 1).sum())
k4.metric("MODIS + VIIRS Co-Located Cells", f"{n_cross}", "Same-day cross-sensor cells")
k5.metric("Peak Cell FRP Density", f"{cells['frpd_harm'].max():.1f} MW/km²", f"Max Power: {cells['frp_harm_mw'].max():.1f} MW")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 Decision Brief (Combat · Prevention · Science)",
    "1. Burning Activity Calendar",
    "2. Cross-Sensor Validation (MODIS vs VIIRS)",
    "3. 2D DIP Raster & Spatial Explorer",
    "4. Scientific Methodology & QA Export",
    "5. Multi-Year Historical Calendar & Alerts"
])

with tab1:
    tz_label = "BRT (UTC-3)" if use_brt_tz else "UTC"
    st.subheader(f"Harmonized Burning Activity Calendar ({tz_label})")
    st.markdown(
        "Compares **raw satellite pixel counts** against **Harmonized Fire Cells** and **Area-Weighted FRP Density ($\\text{MW/km}^2$)**. "
        "Days exceeding $\\mu + 0.75\\sigma$ (computed over complete days, excluding incomplete edge dates) are flagged."
    )

    cal_df = (
        cells.groupby("acq_date", as_index=False)
        .agg(
            Harmonized_Cells=("ix", "count"),
            Raw_Detections=("raw_detections", "sum"),
            Cross_Sensor_Cells=("n_families", lambda x: int((x > 1).sum())),
            Mean_FRP_Density_MW_km2=("frpd_harm", "mean"),
            Total_Harmonized_FRP_MW=("frp_harm_mw", "sum"),
        )
        .sort_values("acq_date")
        .reset_index(drop=True)
    )

    cal_df["Is_Edge_Day"] = False
    if len(cal_df) >= 2:
        cal_df.loc[len(cal_df) - 1, "Is_Edge_Day"] = True
    if use_brt_tz and len(cal_df) >= 3:
        cal_df.loc[0, "Is_Edge_Day"] = True

    ref_df = cal_df[~cal_df["Is_Edge_Day"]] if (exclude_partial_edges and len(cal_df) >= 3) else cal_df
    mean_c = ref_df["Harmonized_Cells"].mean()
    std_c = ref_df["Harmonized_Cells"].std(ddof=0) if len(ref_df) > 1 else 0.0
    thresh = mean_c + 0.75 * std_c

    def classify_day(row):
        if row["Is_Edge_Day"] and exclude_partial_edges:
            return "Partial Edge Day"
        return "Above μ + 0.75σ" if row["Harmonized_Cells"] > thresh else "Normal"

    cal_df["Period_Status"] = cal_df.apply(classify_day, axis=1)
    cal_df["Mean_FRP_Density_MW_km2"] = cal_df["Mean_FRP_Density_MW_km2"].round(2)
    cal_df["Total_Harmonized_FRP_MW"] = cal_df["Total_Harmonized_FRP_MW"].round(1)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_cal, ax1 = plt.subplots(figsize=(8.2, 3.8))
        x = np.arange(len(cal_df["acq_date"]))
        w = 0.36
        ax1.bar(x - w/2, cal_df["Raw_Detections"], w, label="Raw Pixel Detections (MODIS+VIIRS)", color="#f4a261", alpha=0.85)
        ax1.bar(x + w/2, cal_df["Harmonized_Cells"], w, label="Harmonized Fire Cells (Grid)", color="#e63946")
        ax1.axhline(thresh, color="#1d3557", linestyle="--", linewidth=1.2, label=f"Threshold (μ + 0.75σ = {thresh:.1f})")
        ax1.set_xticks(x)
        ax1.set_xticklabels(cal_df["acq_date"], rotation=25, fontsize=8)
        ax1.set_ylabel("Count")
        ax1.set_title(f"Daily Burning Calendar ({tz_label}): Raw Pixels vs. Harmonized Cells")
        ax1.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cal)
        plt.close(fig_cal)

    with c2:
        st.dataframe(cal_df.drop(columns=["Is_Edge_Day"]), use_container_width=True, hide_index=True)

        pivot_hr = d_filt.pivot_table(index="acq_date", columns="hour_plot", values="frp", aggfunc="count", fill_value=0)
        fig_hm, ax_hm = plt.subplots(figsize=(6, 2.3))
        im_hm = ax_hm.imshow(pivot_hr.values, aspect="auto", cmap="YlOrRd")
        step_y = max(1, len(pivot_hr.index) // 8)
        ax_hm.set_yticks(range(0, len(pivot_hr.index), step_y))
        ax_hm.set_yticklabels(pivot_hr.index[::step_y], fontsize=8)
        ax_hm.set_xticks(range(len(pivot_hr.columns)))
        ax_hm.set_xticklabels(pivot_hr.columns, fontsize=7)
        ax_hm.set_xlabel(f"Overpass Hour ({tz_label})", fontsize=8)
        ax_hm.set_title("Diurnal Overpass Persistence Matrix (Hotspots per Hour)", fontsize=9)
        plt.colorbar(im_hm, ax=ax_hm)
        plt.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)

with tab2:
    st.subheader("Cross-Sensor Harmonization Validation by Area Quartiles")
    st.markdown(
        "Evaluates 1-to-1 satellite overpass pairs within $|\\Delta t| \\le \\Delta t_{\\max}$ on a **0.05° (~5.5 km)** grid. "
        "Splitting pairs into area ratio quartiles ($A_{\\text{MODIS}} / A_{\\text{VIIRS}}$) demonstrates how raw FRP scales with pixel size, "
        "whereas area-normalized FRP Density remains stable around $\\approx 0.30\times$."
    )

    c_val1, c_val2 = st.columns(2)
    val_grid = c_val1.selectbox("Validation Grid Resolution (°)", options=[0.05, 0.02, 0.1], index=0)
    max_dt_min = c_val2.slider("Max Overpass Time Diff |Δt| (min)", min_value=15, max_value=360, value=90, step=15)

    paired_view = H.paired_same_overpass(d_filt, grid_res=float(val_grid), max_dt_min=float(max_dt_min), time_col="data_hora_utc")
    sm = H.summarize_pairs(paired_view)

    if sm["n"] == 0:
        st.info(f"No 1-to-1 MODIS + VIIRS pairs found within ±{max_dt_min} min at {val_grid}°.")
    else:
        ci_day = sm.get("ci95_ratio_density_by_day") or sm["ci95_ratio_density"]

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("1-to-1 Independent Pairs (n)", f"{sm['n']}", f"Median |Δt|: {sm['median_dt_min']:.0f} min")
        v2.metric("Median Raw FRP Ratio", f"{sm['median_ratio_raw']:.2f}x", "Extensive (unnormalized)")
        v3.metric("Median Density Ratio", f"{sm['median_ratio_density']:.2f}x", f"Day-Clustered 95% CI: {ci_day[0]:.2f}–{ci_day[1]:.2f}")
        v4.metric("Log10 Dispersion", f"{sm['sd_log10_raw']:.2f} → {sm['sd_log10_density']:.2f}", f"{sm['share_density_below_1']*100:.0f}% VIIRS > MODIS")

        # Calcular Quartis
        q_df = paired_view.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_raw", "ratio_density"]).copy()
        q_df["area_ratio"] = q_df["area_km2_MODIS"] / q_df["area_km2_VIIRS"]
        q_df["quartil"] = pd.qcut(q_df["area_ratio"], 4, labels=["Q1 (Smaller)", "Q2", "Q3", "Q4 (Larger)"])

        res_quartis = q_df.groupby("quartil", observed=True).agg(
            Pairs_n=("ratio_raw", "size"),
            Median_Area_Ratio=("area_ratio", "median"),
            Median_Raw_Ratio=("ratio_raw", "median"),
            Median_Density_Ratio=("ratio_density", "median"),
        ).round(2).reset_index()

        cp1, cp2 = st.columns(2)
        with cp1:
            st.markdown("###Dose-Response Table by Area Quartiles")
            st.dataframe(res_quartis, use_container_width=True, hide_index=True)
            st.caption("Across area-ratio quartiles (1.3× → 15×), the raw MODIS/VIIRS FRP ratio changes ~8-fold, while the area-normalized density ratio changes ~1.7-fold (0.40 → 0.24). Normalization removes most of the pixel-size effect; a residual trend remains, which may partly reflect noise in the measured area ratio. Even at near-equal areas (Q1), VIIRS density is ~2.5× MODIS, indicating a systematic sensor offset not explained by pixel size.")

        with cp2:
            fig_q, ax_q = plt.subplots(figsize=(6, 4.2))
            x_idx = np.arange(len(res_quartis))
            ax_q.plot(x_idx, res_quartis["Median_Raw_Ratio"].to_numpy(), marker="o", color="#e76f51", lw=2, label="Raw FRP Ratio (Extensive)")
            ax_q.plot(x_idx, res_quartis["Median_Density_Ratio"].to_numpy(), marker="s", color="#e63946", lw=2, label="FRP Density Ratio (Intensive)")
            ax_q.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="1:1 Parity")
            ax_q.set_xticks(x_idx)
            ax_q.set_xticklabels(res_quartis["quartil"].astype(str).tolist())
            ax_q.set_ylabel("MODIS / VIIRS Median Ratio")
            ax_q.set_title("Dose-Response: Raw Ratio vs. Area-Normalized Density")
            ax_q.legend(fontsize=8)
            ax_q.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_q)
            plt.close(fig_q)

        st.markdown("###1-to-1 Paired Satellite Observations (`dt_min` sorted)")
        st.dataframe(paired_view.round(2), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🛰️ 1:1 Grid-Aligned Raster & Separable 2D DIP Filtering")
    st.markdown(
        "`build_raster()` maps **1 image pixel = 1 harmonized grid cell** `(ix, iy)`. "
        "Extensive power (`frp_harm_mw`) is summed across days, whereas intensive density (`frpd_harm`) is averaged across active days."
    )
    col_ctrl1, col_ctrl2 = st.columns(2)
    raster_metric = col_ctrl1.selectbox(
        "Raster Cell Value",
        ["frpd_harm", "frp_harm_mw"],
        format_func=lambda x: "Mean FRP Density (MW/km² - Intensive)" if x == "frpd_harm" else "Cumulative Harmonized FRP (MW - Extensive)"
    )
    filter_kind = col_ctrl2.selectbox(
        "2D Spatial Filter (`scipy.ndimage`)",
        ["gaussian", "mean", "log", "laplacian"],
        format_func=lambda x: {
            "gaussian": "Separable Gaussian 3x3 (Smooth Thermal Plume)",
            "mean": "Uniform Box 5x5 (Regional Cluster Density)",
            "log": "Laplacian of Gaussian - LoG (Noise-Robust Fire Fronts)",
            "laplacian": "Signed Laplacian 3x3 (Raw High-Pass Transitions)",
        }[x]
    )

    col_map, col_dip = st.columns(2)
    with col_map:
        st.markdown("**Geospatial Center of Harmonized Fire Cells**")
        map_df = cells.rename(columns={"grid_lat": "latitude", "grid_lon": "longitude"})
        st.map(map_df[["latitude", "longitude"]], size=20, color="#ff4b4b")

    with col_dip:
        img, (x0, y0) = H.build_raster(cells, value=raster_metric, pad=3)
        filtered = H.filter_raster(img, kind=filter_kind)
        extent = [
            x0 * grid_res, (x0 + img.shape[1]) * grid_res,
            y0 * grid_res, (y0 + img.shape[0]) * grid_res,
        ]

        fig_dip, ax_dip = plt.subplots(figsize=(6.2, 4.6))
        if filter_kind in ("laplacian", "log"):
            vmax = np.percentile(np.abs(filtered), 99) if np.any(filtered != 0) else 1.0
            vmax = max(vmax, 1e-3)
            im = ax_dip.imshow(filtered, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=extent, aspect="auto")
            plt.colorbar(im, ax=ax_dip, label="Signed Edge Response")
        else:
            im = ax_dip.imshow(np.log1p(np.maximum(filtered, 0)), origin="lower", cmap="inferno", extent=extent, aspect="auto")
            plt.colorbar(im, ax=ax_dip, label=f"log(1 + {raster_metric})")

        ax_dip.set_xlabel("Longitude (°)")
        ax_dip.set_ylabel("Latitude (°)")
        ax_dip.set_title(f"Grid-Aligned 2D Raster ({img.shape[1]}x{img.shape[0]} cells) — {filter_kind.upper()}")
        plt.tight_layout()
        st.pyplot(fig_dip)
        plt.close(fig_dip)

with tab4:
    st.subheader("Scientific Methodology, Dynamic Empirical Summary & QA Export")
    mod_df = d_filt[d_filt["family"] == "MODIS"]
    vii_df = d_filt[d_filt["family"] == "VIIRS"]
    mod_med_a = mod_df["pixel_area_km2"].median() if not mod_df.empty else np.nan
    vii_med_a = vii_df["pixel_area_km2"].median() if not vii_df.empty else np.nan
    mod_med_d = mod_df["frp_density"].median() if not mod_df.empty else np.nan
    vii_med_d = vii_df["frp_density"].median() if not vii_df.empty else np.nan

    st.markdown(f"""
### 1. Physical Normalization & Dynamic Dataset Summary ($n = {len(d_filt):,}$)
Satellite fire radiative power ($\\text{{FRP}}$, in MW) is an **extensive quantity** integrated over the pixel footprint. In the currently loaded dataset:
* **Observed Pixel Footprint ($\\text{{scan}} \\times \\text{{track}}$):** Median footprint area is **${mod_med_a:.2f}\\text{{ km}}^2$** for MODIS ($n={len(mod_df):,}$) versus **${vii_med_a:.2f}\\text{{ km}}^2$** for VIIRS ($n={len(vii_df):,}$).
* **Observed FRP Density ($\\text{{MW/km}}^2$):** Median area-normalized density is **${mod_med_d:.2f}\\text{{ MW/km}}^2$** for MODIS versus **${vii_med_d:.2f}\\text{{ MW/km}}^2$** for VIIRS. In co-located pairs, VIIRS consistently exhibits higher FRP density ($\\text{{MODIS/VIIRS}} \\approx 0.23\\text{{--}}0.24\\times$). Contributing hypotheses include sub-pixel background cooling dilution in larger MODIS footprints, sensor-specific retrieval differences, overpass timing offsets, and partial footprint overlap.

### 2. NASA FIRMS Data Availability & Documented Limitations
* **VIIRS SNPP NRT Feed Status:** Querying NASA FIRMS `data_availability` confirmed that `VIIRS_SNPP_NRT` stopped updating on `2026-11-02`, whereas `MODIS_NRT`, `VIIRS_NOAA20_NRT`, and `VIIRS_NOAA21_NRT` remain active through `2026-11-14`.
* **Centroid Assignment vs. Footprint Overlap:** Hotspots are assigned to grid cell centers $(i_x + 0.5)\\Delta$ via their geographic centroid. Cells where $\\max(A_i) > A_{{\\text{{cell}}}}$ are flagged via `pixel_gt_cell`.
* **Physical QA Flag (`qa_suspect`):** Flags records with brightness temperature $T < 290\\text{{ K}}$ ({qa.get('qa_suspect', 0)} records in current load) as physically incoherent thermal readings (probable sensor/channel fill artifact).
    """)

    st.markdown("###Export Harmonized Product & QA Summary")
    st.json(qa)
    csv_bytes = cells.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Harmonized Cells CSV (with FRP Density & QA Flags)",
        data=csv_bytes,
        file_name=f"aeroguard_harmonized_{grid_res}deg.csv",
        mime="text/csv",
    )

with tab5:
    history_tab.render(st)

with tab0:
    product_view.render(st, cells=cells)
