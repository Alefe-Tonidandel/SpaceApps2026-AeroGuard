"""
history_tab.py — Aba "Historical Calendar & Alerts" do AeroGuard (Streamlit).

Integração no app.py (3 linhas):
    import history_tab
    ...
    tab_hist = st.tabs([...,"🗓️ 5. Historical Calendar & Alerts"])[-1]   # ou adicione à lista de abas existente
    with tab_hist:
        history_tab.render(st)
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import climatology as C

ARCHIVE = "archive/aeroguard_archive.parquet"
PRESETS = {  # caixas aproximadas, todas dentro da área do arquivo (-51,-23,-39,-14)
    "Entire archive area (MG + Cerrado transition)": (-51.0, -23.0, -39.0, -14.0),
    "Triângulo Mineiro / Alto Paranaíba (approx.)": (-51.0, -20.5, -47.0, -17.5),
    "North of Minas Gerais (approx.)": (-46.0, -18.0, -41.0, -14.0),
    "South of Minas / Mantiqueira (approx.)": (-47.0, -23.0, -42.0, -20.0),
}
METRICS = {
    "Active 0.1° cells per day (recommended)": ("cells", "Active cells / day", "cells_union", "det_raw"),
    "Total FRP per day (MW)": ("frp", "Total FRP (MW) / day", None, "frp_raw"),
}
LEVEL_UI = {  # nível -> (função do streamlit, rótulo)
    "critical": ("error", "🔴 CRITICAL — at or above the historical P90 for this time of year"),
    "elevated": ("warning", "🟠 ELEVATED — between P75 and P90"),
    "normal": ("success", "🟢 NORMAL — within the usual range (P25–P75)"),
    "below": ("info", "🔵 BELOW NORMAL — under the historical P25"),
    "no_data": ("info", "⚪ No data"),
}


def _stamp(path):
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


def render(st, archive_path=ARCHIVE):
    st.subheader("🗓️ Historical Burning Activity Calendar, Unusual Conditions & Alerts")
    if not Path(archive_path).exists() and not Path(archive_path).with_name("aeroguard_archive.csv.gz").exists():
        st.warning(f"Historical archive not found (`{archive_path}`). Run `python3 build_archive.py` first.")
        return

    @st.cache_resource(show_spinner="Loading historical archive...")
    def _load(path, stamp):
        return C.load_archive(path)

    @st.cache_data(show_spinner="Computing series for the selected area...")
    def _series(path, stamp, bbox, metric, mode):
        arch, meta, cov = _load(path, stamp)
        a = C.filter_aoi(arch, bbox, meta.get("cell_deg", 0.1))
        daily = C.daily_metrics(a, cov, full_arch=arch)
        harm, cals, est = C.harmonize_daily(daily, metric=metric, mode=mode)
        cv = {}
        for p in cals:
            t, s = C.cv_calibration(daily, "MODIS", p, metric, mode)
            cv[p] = s
        return daily, harm, cals, cv, int(len(a))

    arch, meta, cov = _load(archive_path, _stamp(archive_path))
    cell = meta.get("cell_deg", 0.1)
    abox = tuple(float(x) for x in meta.get("bbox", "-51,-23,-39,-14").split(","))

    # ------------------------------------------------------------------ AOI + métrica
    c1, c2, c3 = st.columns([2, 2, 1.4])
    mode_aoi = c1.radio("Area of interest", ["Preset", "Custom box", "Point + radius"], horizontal=True)
    if mode_aoi == "Preset":
        name = c1.selectbox("Preset area", list(PRESETS))
        bbox = PRESETS[name]
    elif mode_aoi == "Custom box":
        w = c1.number_input("West (lon)", value=-47.0, min_value=abox[0], max_value=abox[2], step=0.5, format="%.2f")
        e = c1.number_input("East (lon)", value=-44.0, min_value=abox[0], max_value=abox[2], step=0.5, format="%.2f")
        s_ = c2.number_input("South (lat)", value=-20.0, min_value=abox[1], max_value=abox[3], step=0.5, format="%.2f")
        n_ = c2.number_input("North (lat)", value=-17.0, min_value=abox[1], max_value=abox[3], step=0.5, format="%.2f")
        bbox = (min(w, e), min(s_, n_), max(w, e), max(s_, n_))
    else:
        lat = c1.number_input("Center latitude", value=-19.9, min_value=abox[1], max_value=abox[3], step=0.1, format="%.3f")
        lon = c1.number_input("Center longitude", value=-43.9, min_value=abox[0], max_value=abox[2], step=0.1, format="%.3f")
        rad = c2.slider("Radius (km)", 25, 300, 100, step=25)
        bbox = C.bbox_from_radius(lat, lon, rad)
    metric_label = c3.selectbox("Metric", list(METRICS))
    metric, ylabel, union_col, raw_col = METRICS[metric_label]
    calib_mode = c3.selectbox("Era calibration", ["ratio", "loglog"],
                              help="ratio: k = ΣMODIS/ΣVIIRS on overlap days (preserves variance). loglog: log-log OLS (shrinks variance).")
    st.caption(f"Selected box: W {bbox[0]:.2f}, S {bbox[1]:.2f}, E {bbox[2]:.2f}, N {bbox[3]:.2f} "
               f"(archive covers W {abox[0]}, S {abox[1]}, E {abox[2]}, N {abox[3]}; cell = {cell}°). "
               "Calendar days are UTC dates.")

    daily, harm, cals, cv, n_rows = _series(archive_path, _stamp(archive_path), tuple(round(b, 4) for b in bbox), metric, calib_mode)
    if n_rows == 0 or harm.dropna().empty:
        st.warning("No fire activity found inside this box. Try a larger area.")
        return
    daily = daily.copy()
    daily["harm"] = harm
    years = sorted(set(harm.dropna().index.year))
    last_year = years[-1]

    # ------------------------------------------------------------------ alerta para respondedores
    st.markdown("### 1. Where are we compared with a normal year?")
    al = C.alert_summary(harm)
    if al.get("status") in ("no_data", "insufficient_baseline"):
        st.info("Not enough historical data in this area to compare the current period.")
    else:
        kind, text = LEVEL_UI[al["level"]]
        getattr(st, kind)(f"**{text}** — as of {al['date']:%Y-%m-%d}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Percentile vs. same time of year", f"P{al['percentile']:.0f}", f"{al['baseline_years']} baseline years")
        m2.metric("Last 7-day mean", f"{al['value_7d']:.1f}", f"{al['ratio_to_median']:.2f}× the median" if np.isfinite(al["ratio_to_median"]) else None)
        m3.metric("7-day trend", f"{al['trend_ratio']:.2f}×" if np.isfinite(al["trend_ratio"]) else "n/a", "vs. previous 7 days")
        m4.metric("Days ≥ P90 (last 14)", f"{al['days_above_p90_14d']}")
        if al.get("low_activity"):
            st.warning("Low-activity baseline (P90 < 2 per day): percentiles are unstable in small areas or the dry season. Use a larger box before acting on this level.")
        if al["status"] == "low_baseline":
            st.warning("Fewer than 5 baseline years available for this time of year.")
        if al["data_age_days"] > 0:
            st.caption(f"Latest day with data in this area: {al['date']:%Y-%m-%d} ({al['data_age_days']} days before the last archive date).")

    # ------------------------------------------------------------------ calendário + faixa climática
    st.markdown("### 2. Burning activity calendar and historical P10–P90 band")
    cal_col, band_col = st.columns([1.25, 1])
    M = C.calendar_matrix(harm)
    with cal_col:
        fig, ax = plt.subplots(figsize=(8, max(3.2, 0.26 * len(M) + 1.4)))
        C.plot_calendar(ax, M, label=ylabel + " (harmonized)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with band_col:
        year = st.selectbox("Year to compare with history", years[::-1], index=0)
        clim = C.climatology(harm, exclude_years=[year])
        cur = C.smooth(harm).loc[str(year)].dropna()
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        C.plot_band(ax, clim, cur, current_label=str(year), ylabel=ylabel + " (7-day mean)")
        ax.set_title(f"{year} vs. history (baseline excludes {year})", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tbl = C.anomaly_table(harm, year)
        n_crit = int((tbl["level"] == "critical").sum())
        st.caption(f"{n_crit} day(s) of {year} at or above P90 · critical = percentile ≥ 90 within the ±3-day window of all other years.")
        st.download_button("⬇️ Download anomaly table (CSV)", tbl.reset_index().to_csv(index=False).encode(),
                           file_name=f"aeroguard_anomaly_{year}.csv", mime="text/csv")

    # ------------------------------------------------------------------ transição de eras
    st.markdown("### 3. Era transition: MODIS → VIIRS (why raw counts cannot be compared across years)")
    events = C.source_start_events(cov)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    C.plot_eras(ax, daily, harm, metric_raw=raw_col, metric_union=union_col, events=events,
                label_raw=("Raw detections/day (all sensors summed)" if metric == "cells" else "Raw total FRP/day (all sensors summed)"),
                label_harm=f"Harmonized (MODIS-equivalent {ylabel.lower()})")
    ax.set_title("30-day mean: raw multi-sensor records jump when VIIRS enters; the harmonized series does not", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ann = C.annual_table(daily, [raw_col, "harm", f"{metric}_MODIS"])
    steps = []
    for yr, label in [(2012, "VIIRS SNPP enters (2012)"), (2018, "VIIRS NOAA-20 enters (2018)")]:
        if yr in ann.index and (yr - 1) in ann.index and ann.loc[[yr - 1, yr], "complete"].all():
            steps.append({"Transition": label, "Raw (all sensors summed)": C.yoy_step(ann, raw_col, yr),
                          "Harmonized": C.yoy_step(ann, "harm", yr),
                          "MODIS only (control)": C.yoy_step(ann, f"{metric}_MODIS", yr)})
    if steps:
        typical = (ann["harm"] / ann["harm"].shift(1)).loc[ann["complete"] & ann["complete"].shift(1, fill_value=False)]
        st.dataframe(pd.DataFrame(steps).round(2), hide_index=True)
        st.caption("Ratio of the mean daily value in the year of the transition to the previous year. "
                   f"Natural year-to-year variation of the harmonized series in this area ranges {typical.min():.2f}× – {typical.max():.2f}×, "
                   "so a harmonized step inside that range is indistinguishable from normal variability. "
                   "MODIS-only is a homogeneous record (same instruments throughout): if Harmonized and MODIS-only move together at a transition, "
                   "the change reflects real activity rather than an instrument artifact.")

    if cals:
        rows = []
        for p, c in cals.items():
            s = cv.get(p, {})
            rows.append({"VIIRS platform": C.SOURCE_LABEL[p],
                         "Scale k → MODIS" if calib_mode == "ratio" else "log-log slope": c.get("k", c.get("b")),
                         "Overlap days": c["n_days"], "corr (log)": c["corr_log"],
                         "Leave-one-year-out corr": s.get("mean_corr_log"),
                         "Median |annual bias| %": s.get("median_abs_annual_bias_pct")})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    with st.expander("Method & limitations of the era harmonization"):
        st.markdown(
            "- Each VIIRS platform is mapped to the MODIS scale using the days both operated; the harmonized value is the mean of the available estimates.\n"
            "- **Assumption:** the VIIRS↔MODIS relationship is stable over time and across activity levels. The leave-one-year-out check tests this "
            "(a year is held out, MODIS is predicted from VIIRS, and error is reported).\n"
            "- Counting *active cells* (not pixels) removes most of the 375 m vs. 1 km oversampling, but VIIRS still detects smaller fires, hence the calibration.\n"
            "- Days with no data (sensor not yet operating, or archive windows unavailable from the API) are shown as gaps, never as zero fire.\n"
            "- NRT data (recent months) and standard-processing data (older) can differ slightly at the boundary.")
