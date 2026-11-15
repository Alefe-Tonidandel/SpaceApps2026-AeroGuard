"""
climatology.py — Climatologia de atividade de queimadas a partir do arquivo agregado (build_archive.py).

Fluxo:  load_archive -> filter_aoi -> daily_metrics -> harmonize_daily -> climatology / anomaly_table / alert_summary

Ideias centrais
---------------
* Métrica por dia = nº de células ativas (padrão, 0,1° ≈ 11 km) ou FRP total (MW) dentro da AOI.
  Contagens brutas de pixels NÃO servem entre eras: VIIRS (375 m) vê muito mais focos que MODIS (1 km).
* "Sem dado" (sensor não operava / janela faltando) é NaN, nunca zero.
* Harmonização entre eras: cada plataforma VIIRS é calibrada para a ESCALA DO MODIS (série longa e
  homogênea desde 2000) usando os dias de sobreposição; a série harmonizada é a média das estimativas
  disponíveis no dia. A calibração é validada por "leave-one-year-out".
* Anomalia/alerta: percentil do valor atual (média móvel de 7 dias) dentro da distribuição histórica do
  mesmo dia do ano (±3 dias), SEM contar o ano em análise na linha de base.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ["MODIS", "SNPP", "N20", "N21"]
SOURCE_LABEL = {"MODIS": "MODIS (Terra+Aqua)", "SNPP": "VIIRS Suomi-NPP", "N20": "VIIRS NOAA-20", "N21": "VIIRS NOAA-21"}
LEVELS = {"critical": ">= P90", "elevated": "P75–P90", "normal": "P25–P75", "below": "< P25"}


# ----------------------------------------------------------------------------------- carga / AOI
def load_archive(path="archive/aeroguard_archive.parquet"):
    """Retorna (arch, meta, coverage). coverage pode ser None se coverage.csv não existir."""
    p = Path(path)
    if not p.exists() and p.suffix == ".parquet":
        alt = p.with_name("aeroguard_archive.csv.gz")
        p = alt if alt.exists() else p
    arch = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    arch["date"] = pd.to_datetime(arch["date"])
    meta_p = p.with_name("meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"cell_deg": 0.1}
    cov_p = p.with_name("coverage.csv")
    cov = pd.read_csv(cov_p, parse_dates=["start", "end"]) if cov_p.exists() else None
    return arch, meta, cov


def cell_centers(arch, cell):
    return (arch["ix"] + 0.5) * cell, (arch["iy"] + 0.5) * cell


def filter_aoi(arch, bbox, cell=0.1):
    """bbox = (oeste, sul, leste, norte). Seleciona células cujo CENTRO está dentro."""
    w, s, e, n = bbox
    lon, lat = cell_centers(arch, cell)
    return arch[(lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)]


def bbox_from_radius(lat, lon, radius_km):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(np.cos(np.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# ----------------------------------------------------------------------------------- séries diárias
def coverage_days(coverage, sources, idx, arch=None):
    """DataFrame bool (data × fonte): True onde a janela foi baixada com sucesso."""
    cov = pd.DataFrame(False, index=idx, columns=sources)
    if coverage is not None and len(coverage):
        for r in coverage.itertuples():
            if r.source in cov.columns and r.status == "ok":
                cov.loc[max(r.start, idx[0]):min(r.end, idx[-1]), r.source] = True
    elif arch is not None:  # sem coverage.csv: assume [primeira, última] data de cada fonte
        for s in sources:
            d = arch.loc[arch["source"] == s, "date"]
            if len(d):
                cov.loc[d.min():d.max(), s] = True
    return cov


def daily_metrics(arch, coverage=None, full_arch=None):
    """Série diária da AOI. Colunas: cells_<src>, det_<src>, frp_<src>, cells_union, det_raw, frp_raw.
    full_arch (opcional) define o intervalo de datas/fontes quando a AOI está vazia em algum período."""
    ref = full_arch if full_arch is not None else arch
    sources = [s for s in SOURCES if s in set(ref["source"].unique())]
    idx = pd.date_range(ref["date"].min(), ref["date"].max(), freq="D")
    if coverage is not None and len(coverage):
        idx = pd.date_range(min(idx[0], coverage["start"].min()), max(idx[-1], coverage["end"].max()), freq="D")
    cov = coverage_days(coverage, sources, idx, ref)
    out = pd.DataFrame(index=idx)
    for s in sources:
        a = arch[arch["source"] == s]
        g = a.groupby("date").agg(cells=("ix", "size"), det=("n_det", "sum"), frp=("frp_sum", "sum"))
        for m in ("cells", "det", "frp"):
            out[f"{m}_{s}"] = g[m].reindex(idx).fillna(0.0).where(cov[s]) if len(g) else (0.0 * cov[s].astype(float)).where(cov[s])
    u = arch.drop_duplicates(["date", "ix", "iy"]).groupby("date").size().reindex(idx).fillna(0.0)
    out["cells_union"] = u.where(cov.any(axis=1))
    out["det_raw"] = out[[f"det_{s}" for s in sources]].sum(axis=1, min_count=1)
    out["frp_raw"] = out[[f"frp_{s}" for s in sources]].sum(axis=1, min_count=1)
    out.index.name = "date"
    return out


# ----------------------------------------------------------------------------------- calibração entre eras
def apply_calibration(cal, values):
    v = pd.Series(values, dtype=float)
    if cal["mode"] == "ratio":
        return v * cal["k"]
    return np.expm1(cal["a"] + cal["b"] * np.log1p(v)).clip(lower=0)


def fit_calibration(daily, ref="MODIS", target="SNPP", metric="cells", mode="ratio", min_days=60):
    """Mapeia `target` (VIIRS) para a escala de `ref` (MODIS) nos dias de sobreposição.
    mode='ratio': k = Σref/Σtarget (preserva a variância). mode='loglog': OLS em log1p (encolhe a variância)."""
    xc, yc = f"{metric}_{target}", f"{metric}_{ref}"
    if xc not in daily or yc not in daily:
        return None
    m = daily[xc].notna() & daily[yc].notna()
    x, y = daily.loc[m, xc], daily.loc[m, yc]
    if len(x) < min_days or x.sum() <= 0:
        return None
    if mode == "ratio":
        cal = {"mode": "ratio", "k": float(y.sum() / x.sum())}
    else:
        b, a = np.polyfit(np.log1p(x), np.log1p(y), 1)
        cal = {"mode": "loglog", "a": float(a), "b": float(b)}
    pred = apply_calibration(cal, x)
    cal.update(n_days=int(len(x)), corr_log=float(np.corrcoef(np.log1p(pred), np.log1p(y))[0, 1]),
               target=target, ref=ref, metric=metric)
    return cal


def cv_calibration(daily, ref="MODIS", target="SNPP", metric="cells", mode="ratio", min_days=60):
    """Leave-one-year-out: calibra sem o ano Y e prevê o MODIS de Y a partir do VIIRS.
    Retorna (tabela por ano, resumo)."""
    xc, yc = f"{metric}_{target}", f"{metric}_{ref}"
    m = daily[xc].notna() & daily[yc].notna()
    d = daily.loc[m]
    rows = []
    for yr in sorted(set(d.index.year)):
        tr, te = d[d.index.year != yr], d[d.index.year == yr]
        if len(te) < min_days or len(tr) < min_days:
            continue
        cal = fit_calibration(tr, ref, target, metric, mode, min_days)
        if cal is None:
            continue
        pred = apply_calibration(cal, te[xc])
        obs = te[yc]
        rows.append({"year": yr, "n_days": len(te),
                     "corr_log": float(np.corrcoef(np.log1p(pred), np.log1p(obs))[0, 1]),
                     "annual_bias_pct": float(100 * (pred.sum() / max(obs.sum(), 1e-9) - 1)),
                     "rmse_log": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2)))})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, {}
    return t, {"mean_corr_log": float(t.corr_log.mean()), "median_abs_annual_bias_pct": float(t.annual_bias_pct.abs().median()),
               "mean_rmse_log": float(t.rmse_log.mean()), "years": int(len(t))}


def harmonize_daily(daily, metric="cells", ref="MODIS", platforms=("SNPP", "N20", "N21"), mode="ratio"):
    """Série harmonizada na escala do MODIS = média das estimativas disponíveis no dia.
    Retorna (harm: Series, calibrations: dict, estimates: DataFrame)."""
    est = {ref: daily[f"{metric}_{ref}"]} if f"{metric}_{ref}" in daily else {}
    cals = {}
    for p in platforms:
        cal = fit_calibration(daily, ref, p, metric, mode)
        if cal is not None:
            cals[p] = cal
            est[p] = apply_calibration(cal, daily[f"{metric}_{p}"])
    E = pd.DataFrame(est)
    harm = E.mean(axis=1, skipna=True)
    harm.name = f"harm_{metric}"
    return harm, cals, E


def annual_table(df, cols, min_coverage=0.9):
    """Média diária por ano (robusta a anos parciais) + fração de dias com dado."""
    g = df[cols].groupby(df.index.year)
    out = g.mean()
    ndays = pd.Series({y: 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365 for y in out.index})
    cover = g.count().min(axis=1) / ndays
    out["coverage"] = cover
    out["complete"] = cover >= min_coverage
    out.index.name = "year"
    return out


def yoy_step(annual, col, year):
    """Razão média_diária(ano)/média_diária(ano-1) — compare com a variação típica entre anos."""
    if year in annual.index and (year - 1) in annual.index and annual.loc[year - 1, col] > 0:
        return float(annual.loc[year, col] / annual.loc[year - 1, col])
    return np.nan


# ----------------------------------------------------------------------------------- climatologia
def doy365(idx):
    """Dia do ano em calendário de 365 dias (29/fev conta junto com 28/fev)."""
    idx = pd.DatetimeIndex(idx)
    d = idx.dayofyear.to_numpy().copy()
    leap = np.asarray(idx.is_leap_year)
    d = np.where(leap & (idx.month > 2), d - 1, d)
    d = np.where(leap & (idx.month == 2) & (idx.day == 29), 59, d)
    return d


def smooth(series, window=7):
    return series.rolling(window, min_periods=max(3, window // 2)).mean()


def doy_matrix(series):
    s = series.dropna()
    df = pd.DataFrame({"year": s.index.year, "doy": doy365(s.index), "v": s.values})
    M = df.pivot_table(index="year", columns="doy", values="v", aggfunc="mean")
    return M.reindex(columns=range(1, 366))


def _pool(vals, d0, hw):
    cols = [(d0 + k) % 365 for k in range(-hw, hw + 1)]
    p = vals[:, cols].ravel()
    return p[~np.isnan(p)]


def climatology(series, exclude_years=(), half_window=3, min_values=15, qs=(10, 25, 50, 75, 90)):
    """Quantis por dia do ano (1..365), agrupando ±half_window dias de TODOS os anos da linha de base."""
    M = doy_matrix(series).drop(index=list(exclude_years), errors="ignore")
    vals = M.to_numpy()
    rows = []
    for d0 in range(365):
        p = _pool(vals, d0, half_window)
        if len(p) >= min_values:
            rows.append([*np.percentile(p, qs), p.mean(), len(p)])
        else:
            rows.append([np.nan] * (len(qs) + 1) + [len(p)])
    out = pd.DataFrame(rows, columns=[f"p{q}" for q in qs] + ["mean", "n"], index=range(1, 366))
    out.index.name = "doy"
    return out


def level_from_percentile(p):
    if np.isnan(p):
        return "no_data"
    return "critical" if p >= 90 else "elevated" if p >= 75 else "normal" if p >= 25 else "below"


def anomaly_table(series, year, window=7, half_window=3, baseline_years=None, min_values=15):
    """Para cada dia de `year`: valor suavizado, P10/P50/P90 históricos, percentil e nível.
    A linha de base exclui `year` (e usa só baseline_years, se dado)."""
    s = smooth(series, window)
    M = doy_matrix(s)
    M = M.drop(index=year, errors="ignore")
    if baseline_years is not None:
        M = M.loc[M.index.intersection(list(baseline_years))]
    vals = M.to_numpy()
    cur = s[s.index.year == year].dropna()
    rows = []
    for date, v in cur.items():
        d0 = int(doy365([date])[0]) - 1
        p = _pool(vals, d0, half_window)
        if len(p) < min_values:
            rows.append([date, v, np.nan, np.nan, np.nan, np.nan, np.nan, len(p)])
            continue
        pct = 100 * ((p < v).mean() + 0.5 * (p == v).mean())
        rows.append([date, v, *np.percentile(p, [10, 50, 90]), pct, level_from_percentile(pct), len(p)])
    t = pd.DataFrame(rows, columns=["date", "value", "p10", "p50", "p90", "percentile", "level", "n_pool"]).set_index("date")
    t["level"] = t["level"].where(t["percentile"].notna(), "no_data")
    return t


def alert_summary(series, as_of=None, window=7, min_years=5):
    """Resumo para respondedores: nível, percentil, razão à mediana, tendência (7d vs 7d anteriores)."""
    valid = series.dropna()
    if valid.empty:
        return {"status": "no_data"}
    last = valid.index.max()
    as_of = min(pd.Timestamp(as_of), last) if as_of is not None else last
    tbl = anomaly_table(series, as_of.year, window)
    tbl = tbl.loc[:as_of]
    tbl = tbl[tbl["percentile"].notna()]
    if tbl.empty:
        return {"status": "insufficient_baseline", "as_of": as_of}
    row = tbl.iloc[-1]
    n_years = int(round(row["n_pool"] / (2 * 3 + 1)))
    recent = series.loc[:as_of].dropna()
    t_now, t_prev = recent.iloc[-7:].mean(), recent.iloc[-14:-7].mean()
    return {
        "status": "ok" if n_years >= min_years else "low_baseline",
        "date": row.name, "as_of": as_of, "data_age_days": int((as_of - row.name).days),
        "level": row["level"], "percentile": float(row["percentile"]),
        "value_7d": float(row["value"]), "p50": float(row["p50"]), "p90": float(row["p90"]),
        "ratio_to_median": float(row["value"] / row["p50"]) if row["p50"] > 0 else np.nan,
        "trend_ratio": float(t_now / t_prev) if t_prev and t_prev > 0 else np.nan,
        "days_above_p90_14d": int((tbl.iloc[-14:]["percentile"] >= 90).sum()),
        "baseline_years": n_years,
        "low_activity": bool(row["p90"] < 2),   # AOI pequena/época seca: percentis ficam instáveis
    }


# ----------------------------------------------------------------------------------- gráficos
_MONTH_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_LBL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def calendar_matrix(series, smooth_window=7, years=None):
    M = doy_matrix(smooth(series, smooth_window))
    return M if years is None else M.loc[M.index.intersection(list(years))]


def plot_calendar(ax, M, label="Active cells / day (7-day mean)", log=True):
    import matplotlib.pyplot as plt
    arr = np.ma.masked_invalid(M.to_numpy(dtype=float))
    vis = np.log1p(arr) if log else arr
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d9d9d9")  # cinza = sem dado
    im = ax.imshow(vis, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xticks([d - 1 for d in _MONTH_DOY])
    ax.set_xticklabels(_MONTH_LBL, fontsize=8)
    ax.set_title("Burning activity calendar — year × day-of-year (grey = no data)", fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(("log(1 + " if log else "") + label + (")" if log else ""), fontsize=8)
    return im


def plot_band(ax, clim, current=None, current_label="Current year", ylabel="Active cells / day (7-day mean)"):
    x = clim.index.to_numpy()
    ax.fill_between(x, clim["p10"].to_numpy(), clim["p90"].to_numpy(), color="#f4a261", alpha=0.35, label="P10–P90 (history)")
    ax.fill_between(x, clim["p25"].to_numpy(), clim["p75"].to_numpy(), color="#e76f51", alpha=0.35, label="P25–P75")
    ax.plot(x, clim["p50"].to_numpy(), color="#9d0208", lw=1.4, label="Median")
    if current is not None and len(current):
        ax.plot(np.asarray(doy365(current.index)), current.to_numpy(), color="#1d3557", lw=1.8, label=current_label)
    ax.set_xticks(_MONTH_DOY)
    ax.set_xticklabels(_MONTH_LBL, fontsize=8)
    ax.set_xlim(1, 365)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper left")


def plot_eras(ax, daily, harm, metric_raw="det_raw", metric_union="cells_union", events=None, roll=30,
              label_raw="Raw detections/day (all sensors summed)", label_harm="Harmonized (MODIS-equivalent)"):
    r = lambda s: s.rolling(roll, min_periods=roll // 2).mean()
    idx_np = daily.index.to_numpy()
    ax.plot(idx_np, r(daily[metric_raw]).to_numpy(), color="#e63946", lw=1.1, label=label_raw)
    if metric_union:
        ax.plot(idx_np, r(daily[metric_union]).to_numpy(), color="#f4a261", lw=1.1, label="Active cells/day, union (no calibration)")
    ax.plot(harm.index.to_numpy(), r(harm).to_numpy(), color="#1d3557", lw=1.6, label=label_harm)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel(f"{roll}-day mean")
    for date, name in (events or {}).items():
        ax.axvline(pd.Timestamp(date), color="k", ls="--", lw=0.8)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1], " " + name, fontsize=7, va="top", rotation=90)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)


def source_start_events(coverage, sources=("SNPP", "N20", "N21")):
    """Datas de início de cada plataforma VIIRS (primeira janela baixada) para marcar nos gráficos."""
    ev = {}
    if coverage is None or coverage.empty:
        return ev
    for s in sources:
        c = coverage[coverage["source"] == s]
        if len(c):
            ev[c["start"].min().strftime("%Y-%m-%d")] = SOURCE_LABEL[s]
    return ev
