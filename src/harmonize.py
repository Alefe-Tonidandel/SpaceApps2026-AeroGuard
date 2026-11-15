"""
harmonize.py — Harmonização MODIS + VIIRS (NASA FIRMS) com normalização por área real de pixel.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

KM_PER_DEG = 111.32
REQUIRED = ["latitude", "longitude", "acq_date", "frp", "confidence", "sensor_source"]
NOMINAL_AREA_KM2 = {"MODIS": 1.0, "VIIRS": 0.375 * 0.375}
VIIRS_CONF = {"l": "low", "n": "nominal", "h": "high"}


def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    d = df.copy()
    n0 = len(d)
    for c in ["latitude", "longitude", "frp", "scan", "track", "bright_temp_k"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "frp"])
    d = d[d["latitude"].between(-90, 90) & d["longitude"].between(-180, 180) & (d["frp"] >= 0)]
    d["acq_date"] = d["acq_date"].astype(str)

    d["family"] = np.where(d["sensor_source"].str.startswith("MODIS"), "MODIS", "VIIRS")

    scan = d["scan"] if "scan" in d.columns else pd.Series(np.nan, index=d.index)
    track = d["track"] if "track" in d.columns else pd.Series(np.nan, index=d.index)
    area = scan * track
    bad_area = ~(area > 0)
    area = area.where(~bad_area, d["family"].map(NOMINAL_AREA_KM2))
    d["pixel_area_km2"] = area
    d["area_is_fallback"] = bad_area
    d["frp_density"] = d["frp"] / d["pixel_area_km2"]

    conf = d["confidence"]
    num = pd.to_numeric(conf, errors="coerce")
    modis_cls = pd.cut(num, [-1, 29, 79, 100], labels=["low", "nominal", "high"]).astype(object)
    viirs_cls = conf.astype(str).str.lower().map(VIIRS_CONF)
    d["conf_class"] = np.where(d["family"] == "MODIS", modis_cls, viirs_cls)
    d["conf_class"] = d["conf_class"].fillna("nominal")

    d["qa_suspect"] = False
    if "bright_temp_k" in d.columns:
        d["qa_suspect"] = d["bright_temp_k"] < 290

    qa = {
        "rows_in": n0,
        "rows_dropped_invalid": n0 - len(d),
        "area_fallback": int(d["area_is_fallback"].sum()),
        "qa_suspect": int(d["qa_suspect"].sum()),
    }
    return d, qa


def cell_area_km2(lat_center, grid_res):
    return (grid_res * KM_PER_DEG) ** 2 * np.cos(np.radians(lat_center))


def harmonize(d: pd.DataFrame, grid_res: float = 0.02):
    x = d.copy()
    x["ix"] = np.floor(x["longitude"] / grid_res).astype(int)
    x["iy"] = np.floor(x["latitude"] / grid_res).astype(int)

    key = ["acq_date", "ix", "iy", "family", "sensor_source"]
    per = x.groupby(key, as_index=False).agg(
        frp_sum=("frp", "sum"),
        area_km2=("pixel_area_km2", "sum"),
        n_pixels=("frp", "size"),
        pixel_area_max=("pixel_area_km2", "max"),
    )
    per["frpd"] = per["frp_sum"] / per["area_km2"]

    g = per.groupby(["acq_date", "ix", "iy"])
    cells = g.agg(
        frp_harm_mw=("frp_sum", "max"),
        frp_sum_all=("frp_sum", "sum"),
        area_sum_all=("area_km2", "sum"),
        raw_detections=("n_pixels", "sum"),
        n_sources=("sensor_source", "nunique"),
        n_families=("family", "nunique"),
        pixel_area_max=("pixel_area_max", "max"),
        sensors=("sensor_source", lambda s: "+".join(sorted(set(s)))),
    ).reset_index()
    cells["frpd_harm"] = cells["frp_sum_all"] / cells["area_sum_all"]
    cells["grid_lon"] = (cells["ix"] + 0.5) * grid_res
    cells["grid_lat"] = (cells["iy"] + 0.5) * grid_res
    cells["cell_km2"] = cell_area_km2(cells["grid_lat"], grid_res)
    cells["pixel_gt_cell"] = cells["pixel_area_max"] > cells["cell_km2"]
    return cells, per


def paired_frp(per: pd.DataFrame) -> pd.DataFrame:
    p = per.pivot_table(index=["acq_date", "ix", "iy"], columns="family",
                        values=["frp_sum", "frpd"], aggfunc="max")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.dropna().reset_index()
    if not p.empty:
        p["ratio_raw"] = p["frp_sum_MODIS"] / p["frp_sum_VIIRS"]
        p["ratio_density"] = p["frpd_MODIS"] / p["frpd_VIIRS"]
    return p


def build_raster(cells: pd.DataFrame, value: str = "frp_harm_mw", pad: int = 3, agg: str | None = None):
    agg = agg or ("sum" if value == "frp_harm_mw" else "mean")
    ix, iy = cells["ix"].to_numpy(), cells["iy"].to_numpy()
    x0, y0 = ix.min() - pad, iy.min() - pad
    shape = (iy.max() - y0 + pad + 1, ix.max() - x0 + pad + 1)
    img = np.zeros(shape, dtype=np.float32)
    np.add.at(img, (iy - y0, ix - x0), cells[value].to_numpy(dtype=np.float32))
    if agg == "mean":
        cnt = np.zeros(shape, dtype=np.float32)
        np.add.at(cnt, (iy - y0, ix - x0), 1.0)
        img = np.divide(img, cnt, out=np.zeros_like(img), where=cnt > 0)
    return img, (x0, y0)


_K1D = np.array([1, 2, 1], dtype=np.float32) / 4
_LAP8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)


def filter_raster(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "gaussian":
        out = ndimage.convolve1d(img, _K1D, axis=0, mode="reflect")
        return ndimage.convolve1d(out, _K1D, axis=1, mode="reflect")
    if kind == "mean":
        return ndimage.uniform_filter(img, size=5, mode="reflect")
    if kind == "laplacian":
        return ndimage.convolve(img, _LAP8, mode="reflect")
    if kind == "log":
        return -ndimage.gaussian_laplace(img, sigma=1.0, mode="reflect")
    raise ValueError(kind)


def paired_same_overpass(d: pd.DataFrame, grid_res: float = 0.05, max_dt_min: float = 90,
                         time_col: str = "data_hora_utc", nearest_only: bool = True) -> pd.DataFrame:
    x = d.copy()
    if time_col not in x.columns or x[time_col].isna().all():
        x[time_col] = pd.to_datetime(
            x["acq_date"].astype(str) + " " + x["acq_time"].astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M", errors="coerce"
        )
    x = x.dropna(subset=[time_col]).copy()
    x[time_col] = pd.to_datetime(x[time_col])
    x["ix"] = np.floor(x["longitude"] / grid_res).astype(int)
    x["iy"] = np.floor(x["latitude"] / grid_res).astype(int)
    x["utc_day"] = x[time_col].dt.floor("D")
    sat = "satellite" if "satellite" in x.columns else "sensor_source"
    dn = "daynight" if "daynight" in x.columns else None
    key = ["family", sat, "ix", "iy", "utc_day"] + ([dn] if dn else [])
    a = x.groupby(key, as_index=False).agg(
        t=(time_col, "mean"), frp_sum=("frp", "sum"), area_km2=("pixel_area_km2", "sum"), n_pixels=("frp", "size"))
    a["frpd"] = a["frp_sum"] / a["area_km2"]
    m, v = a[a.family == "MODIS"], a[a.family == "VIIRS"]
    p = m.merge(v, on=["ix", "iy"], suffixes=("_MODIS", "_VIIRS"))
    p["dt_min"] = (p["t_MODIS"] - p["t_VIIRS"]).abs().dt.total_seconds() / 60
    p = p[p["dt_min"] <= max_dt_min].copy()
    if nearest_only and not p.empty:
        p = p.sort_values("dt_min")
        kM = ["ix", "iy", f"{sat}_MODIS", "utc_day_MODIS"] + ([f"{dn}_MODIS"] if dn else [])
        kV = ["ix", "iy", f"{sat}_VIIRS", "utc_day_VIIRS"] + ([f"{dn}_VIIRS"] if dn else [])
        p = p.drop_duplicates(kM).drop_duplicates(kV).copy()
    p["ratio_raw"] = p["frp_sum_MODIS"] / p["frp_sum_VIIRS"]
    p["ratio_density"] = p["frpd_MODIS"] / p["frpd_VIIRS"]
    cols = ["ix", "iy", "utc_day_MODIS", f"{sat}_MODIS", f"{sat}_VIIRS", "dt_min", "n_pixels_MODIS", "n_pixels_VIIRS",
            "area_km2_MODIS", "area_km2_VIIRS",
            "frp_sum_MODIS", "frp_sum_VIIRS", "frpd_MODIS", "frpd_VIIRS", "ratio_raw", "ratio_density"]
    return p[cols].sort_values("dt_min").reset_index(drop=True)


def summarize_pairs(p: pd.DataFrame, n_boot: int = 2000, seed: int = 0, cluster: str = "utc_day_MODIS") -> dict:
    if p is None or p.empty:
        return {"n": 0}
    q = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_raw", "ratio_density"])
    q = q[(q.ratio_raw > 0) & (q.ratio_density > 0)]
    if q.empty:
        return {"n": 0}
    r = q["ratio_density"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(r, size=(n_boot, len(r))), axis=1)
    ci_cluster = None
    if cluster in q.columns and q[cluster].nunique() >= 5:
        groups = [g["ratio_density"].to_numpy() for _, g in q.groupby(cluster)]
        bc = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
              for _ in range(n_boot)]
        ci_cluster = (float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5)))
    out = {
        "n": int(len(q)),
        "median_dt_min": float(q["dt_min"].median()),
        "median_ratio_raw": float(q["ratio_raw"].median()),
        "median_ratio_density": float(np.median(r)),
        "ci95_ratio_density": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "ci95_ratio_density_by_day": ci_cluster,
        "sd_log10_raw": float(np.log10(q["ratio_raw"]).std(ddof=1)) if len(q) > 1 else np.nan,
        "sd_log10_density": float(np.log10(q["ratio_density"]).std(ddof=1)) if len(q) > 1 else np.nan,
        "share_density_below_1": float((r < 1).mean()),
    }
    if len(q) >= 4:
        out["spearman_frp"] = float(spearmanr(q["frp_sum_MODIS"], q["frp_sum_VIIRS"])[0])
        out["spearman_density"] = float(spearmanr(q["frpd_MODIS"], q["frpd_VIIRS"])[0])
    return out


def area_dependence(p: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    q = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_raw", "ratio_density", "area_km2_MODIS", "area_km2_VIIRS"])
    q = q[(q.ratio_raw > 0) & (q.ratio_density > 0)]
    if len(q) < 10:
        return {"n": int(len(q))}
    x = np.log10(q["area_km2_MODIS"] / q["area_km2_VIIRS"]).to_numpy()
    out = {"n": int(len(q))}
    rng = np.random.default_rng(seed)
    for name, y in (("raw", np.log10(q["ratio_raw"]).to_numpy()), ("density", np.log10(q["ratio_density"]).to_numpy())):
        slope = np.polyfit(x, y, 1)[0]
        idx = rng.integers(0, len(x), (n_boot, len(x)))
        bs = [np.polyfit(x[i], y[i], 1)[0] for i in idx]
        out[f"slope_{name}"] = float(slope)
        out[f"slope_{name}_ci95"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out
