#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
"""
build_archive.py — Arquivo histórico agregado MODIS/VIIRS (NASA FIRMS) com checkpoint e retomada.

Para cada janela de até 5 dias e cada sensor (MODIS, VIIRS SNPP, VIIRS NOAA-20, opcional NOAA-21):
  baixa os focos (SP = processamento padrão para o passado, NRT para o recente), filtra
  (confiança != low, T >= 290 K, type == 0 quando existir), agrega por (data, célula, sensor) e grava
  uma "parte" em archive/parts/. Rodar de novo pula as partes que já existem (retomada).
  No fim, junta tudo em archive/aeroguard_archive.parquet (+ coverage.csv + meta.json).

Uso rápido:
  python3 build_archive.py --dry-run                      # mostra o plano e a estimativa, sem baixar
  python3 build_archive.py                                # 2010-01-01 até o fim da disponibilidade
  python3 build_archive.py --months 6-11 --start 2015-01-01
  python3 build_archive.py --merge-only                   # só reagrega as partes já baixadas

Chave: variável de ambiente FIRMS_KEY ou .streamlit/secrets.toml (FIRMS_KEY = "...").
"""
import argparse
import datetime as dt
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://firms.modaps.eosdis.nasa.gov/api"
DEFAULT_BBOX = "-51,-23,-39,-14"          # oeste,sul,leste,norte (MG + transição Cerrado)
FAMILIES = {                              # família -> (fonte SP, fonte NRT)
    "MODIS": ("MODIS_SP", "MODIS_NRT"),
    "SNPP": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
    "N20": ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    "N21": ("VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"),
}
DEFAULT_SOURCES = ["MODIS", "SNPP", "N20"]   # N21 só adiciona uma plataforma quase idêntica à N20
NOMINAL_AREA_KM2 = {"MODIS": 1.0, "SNPP": 0.375 ** 2, "N20": 0.375 ** 2, "N21": 0.375 ** 2}
MAX_RANGE = 5
ARCHIVE_COLS = ["date", "ix", "iy", "source", "n_det", "frp_sum", "area_km2"]


class FatalAPIError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------- chave
def read_key() -> str:
    key = os.environ.get("FIRMS_KEY", "").strip()
    p = Path(".streamlit/secrets.toml")
    if not key and p.exists():  # sem tomllib (Python 3.10): leitura simples
        m = re.search(r'FIRMS_KEY\s*=\s*["\']?\s*([A-Za-z0-9]+)', p.read_text(encoding="utf-8"))
        key = m.group(1) if m else ""
    return key


# ----------------------------------------------------------------------------------- HTTP
_rl_lock = threading.Lock()
_resume_at = 0.0


def _set_pause(seconds: float):
    global _resume_at
    with _rl_lock:
        _resume_at = max(_resume_at, time.time() + seconds)


def http_get(url: str, retries: int = 5, timeout: int = 90):
    """Retorna o texto (str, possivelmente vazio) ou None se a janela falhou.
    Levanta FatalAPIError para chave inválida. Limite de transações: pausa TODAS as threads e tenta de novo."""
    last = ""
    for attempt in range(1, retries + 1):
        wait = _resume_at - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(min(2 ** attempt, 30))
            continue
        txt = (r.text or "").strip()
        low = txt.lower()
        if r.status_code == 200 and (txt.startswith(("latitude", "data_id")) or txt == ""):
            return txt
        if "exceed" in low or "transaction" in low or "invalid map_key" in low or "invalid key" in low or r.status_code == 429:
            pause = min(120 * attempt, 660)
            print(f"  [limite de transações] pausando {pause}s...", flush=True)
            _set_pause(pause)
            last = txt[:100]
            continue
        if r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(min(2 ** attempt, 30))
            continue
        last = txt[:120] or f"HTTP {r.status_code}"
        break  # erro determinístico (data inválida etc.): não insistir
    print(f"  [falhou] {url.split('/area/csv/')[-1].split('/', 1)[-1]}: {last}", flush=True)
    return None


def get_availability(key: str) -> dict:
    """{data_id: (min_date, max_date)} lido ao vivo do endpoint data_availability."""
    txt = http_get(f"{BASE}/data_availability/csv/{key}/ALL")
    if not txt:
        raise SystemExit("Não consegui ler data_availability (verifique a chave e a conexão).")
    df = pd.read_csv(io.StringIO(txt))
    out = {}
    for r in df.itertuples():
        out[r.data_id] = (pd.Timestamp(r.min_date).date(), pd.Timestamp(r.max_date).date())
    return out


# ----------------------------------------------------------------------------------- plano
@dataclass(frozen=True)
class Window:
    family: str
    src: str
    start: dt.date
    days: int

    @property
    def end(self):
        return self.start + dt.timedelta(days=self.days - 1)

    @property
    def wid(self):
        return f"{self.family}_{self.src}_{self.start:%Y%m%d}_{self.days}d"


def parse_months(s: str) -> set:
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def plan_windows(fam: str, avail: dict, start: dt.date, end: dt.date, months: set):
    """Cada data recebe SP (se disponível) ou NRT; datas sem fonte viram lacuna. Janelas de <= 5 dias contíguos."""
    sp, nrt = FAMILIES[fam]
    ranges = [(n, avail[n]) for n in (sp, nrt) if n in avail]
    if not ranges:
        return [], []
    first = min(r[0] for _, r in ranges)
    last = max(r[1] for _, r in ranges)
    end = min(end, last)
    windows, gaps, cur = [], [], None
    d = start
    while d <= end:
        src = next((n for n, (a, b) in ranges if a <= d <= b), None)
        if d.month not in months:
            cur = None
        elif src is None:
            cur = None
            if d >= first:  # lacuna interna (antes do início do sensor não conta como lacuna)
                if gaps and gaps[-1][1] == d - dt.timedelta(days=1):
                    gaps[-1] = (gaps[-1][0], d)
                else:
                    gaps.append((d, d))
        else:
            if cur and cur["src"] == src and cur["n"] < MAX_RANGE and cur["last"] == d - dt.timedelta(days=1):
                cur["n"] += 1
                cur["last"] = d
            else:
                cur = {"src": src, "start": d, "n": 1, "last": d}
                windows.append(cur)
        d += dt.timedelta(days=1)
    return [Window(fam, w["src"], w["start"], w["n"]) for w in windows], gaps


# ----------------------------------------------------------------------------------- agregação
def aggregate(df: pd.DataFrame, fam: str, cell: float) -> pd.DataFrame:
    """Filtra e agrega focos brutos por (data, célula, sensor). Colunas = ARCHIVE_COLS."""
    if df is None or df.empty:
        return pd.DataFrame(columns=ARCHIVE_COLS)
    d = df.copy()
    for c in ["latitude", "longitude", "frp", "scan", "track", "brightness", "bright_ti4"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "frp", "acq_date"])
    d = d[d["frp"] >= 0]

    if "type" in d.columns:  # exclui vulcões, fontes estáticas e offshore (mantém vegetação = 0)
        t = pd.to_numeric(d["type"], errors="coerce")
        d = d[t.isna() | (t == 0)]

    conf = d["confidence"] if "confidence" in d.columns else pd.Series("n", index=d.index)
    num = pd.to_numeric(conf, errors="coerce")
    cls = pd.Series(np.where(num.notna(), np.where(num < 30, "low", "ok"), ""), index=d.index)
    txt = conf.astype(str).str.lower().str.strip()
    cls = cls.where(num.notna(), np.where(txt == "l", "low", "ok"))
    d = d[cls != "low"]

    bt = d["brightness"] if "brightness" in d.columns else d.get("bright_ti4")
    if bt is not None:
        d = d[bt.isna() | (bt >= 290)]
    if d.empty:
        return pd.DataFrame(columns=ARCHIVE_COLS)

    scan = d["scan"] if "scan" in d.columns else pd.Series(np.nan, index=d.index)
    track = d["track"] if "track" in d.columns else pd.Series(np.nan, index=d.index)
    area = (scan * track).where((scan * track) > 0, NOMINAL_AREA_KM2[fam])
    d = d.assign(
        date=pd.to_datetime(d["acq_date"].astype(str)).dt.strftime("%Y-%m-%d"),
        ix=np.floor(d["longitude"] / cell).astype(int),
        iy=np.floor(d["latitude"] / cell).astype(int),
        area=area,
    )
    g = d.groupby(["date", "ix", "iy"], as_index=False).agg(
        n_det=("frp", "size"), frp_sum=("frp", "sum"), area_km2=("area", "sum"))
    g["source"] = fam
    g["frp_sum"] = g["frp_sum"].round(2).astype("float32")
    g["area_km2"] = g["area_km2"].round(3).astype("float32")
    return g[ARCHIVE_COLS]


# ----------------------------------------------------------------------------------- execução
def part_path(outdir: Path, w: Window) -> Path:
    return outdir / "parts" / f"{w.wid}.csv.gz"


def run_window(w: Window, key: str, bbox: str, cell: float, outdir: Path):
    url = f"{BASE}/area/csv/{key}/{w.src}/{bbox}/{w.days}/{w.start.isoformat()}"
    txt = http_get(url)
    if txt is None:
        return ("failed", w, 0, 0)
    raw = pd.read_csv(io.StringIO(txt)) if txt else pd.DataFrame()
    agg = aggregate(raw, w.family, cell)
    p = part_path(outdir, w)
    tmp = p.with_name(p.name + ".tmp")
    agg.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, p)  # gravação atômica: parte incompleta nunca conta como feita
    return ("ok", w, len(raw), len(agg))


def merge(outdir: Path, windows: list, bbox: str, cell: float):
    """Junta SOMENTE as partes do plano atual (evita contagem dupla se o plano mudou entre execuções)."""
    frames, cov, missing = [], [], 0
    for w in windows:
        p = part_path(outdir, w)
        if not p.exists():
            missing += 1
            continue
        df = pd.read_csv(p, dtype={"date": str})
        if len(df):
            frames.append(df)
        cov.append({"source": w.family, "start": w.start.isoformat(), "end": w.end.isoformat(),
                    "api_source": w.src, "status": "ok"})
    arch = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ARCHIVE_COLS)
    arch = arch.drop_duplicates(["date", "ix", "iy", "source"]).sort_values(["date", "source", "iy", "ix"])
    arch["ix"] = arch["ix"].astype("int32")
    arch["iy"] = arch["iy"].astype("int32")
    arch["n_det"] = arch["n_det"].astype("int32")
    try:
        arch.to_parquet(outdir / "aeroguard_archive.parquet", index=False)
        fmt = "parquet"
    except Exception as e:  # sem pyarrow
        print(f"  (parquet indisponível: {e}); gravando csv.gz")
        arch.to_csv(outdir / "aeroguard_archive.csv.gz", index=False, compression="gzip")
        fmt = "csv.gz"
    pd.DataFrame(cov).to_csv(outdir / "coverage.csv", index=False)
    meta = {"bbox": bbox, "cell_deg": cell, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(arch)), "format": fmt, "windows_ok": len(cov), "windows_missing": missing,
            "date_range": [arch["date"].min(), arch["date"].max()] if len(arch) else None}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return arch, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", default=DEFAULT_BBOX, help="oeste,sul,leste,norte")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="MODIS,SNPP,N20,N21")
    ap.add_argument("--months", default="1-12", help="ex.: 1-12 | 6-11 | 6,7,8")
    ap.add_argument("--cell", type=float, default=0.1, help="tamanho da célula em graus")
    ap.add_argument("--out", default="archive")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="limita a qtde de janelas (teste)")
    a = ap.parse_args(argv)

    outdir = Path(a.out)
    (outdir / "parts").mkdir(parents=True, exist_ok=True)
    key = read_key()
    if not key:
        raise SystemExit("Defina FIRMS_KEY (variável de ambiente ou .streamlit/secrets.toml).")

    avail = get_availability(key)
    start, end, months = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), parse_months(a.months)
    windows, gaps_all = [], {}
    for fam in [s.strip().upper() for s in a.sources.split(",")]:
        w, gaps = plan_windows(fam, avail, start, end, months)
        windows += w
        gaps_all[fam] = gaps
    todo = [w for w in windows if not part_path(outdir, w).exists()]

    print(f"Plano: {len(windows)} janelas | já feitas: {len(windows) - len(todo)} | a baixar: {len(todo)}")
    for fam in sorted({w.family for w in windows}):
        n = sum(w.family == fam for w in windows)
        rng = [w for w in windows if w.family == fam]
        print(f"  {fam:5s} {n:5d} janelas  {rng[0].start} → {rng[-1].end}")
        for g0, g1 in gaps_all.get(fam, []):
            print(f"        ⚠ lacuna de disponibilidade: {g0} → {g1} ({(g1 - g0).days + 1} dias) — ficará SEM dados")
    est_min = len(todo) * 2.0 / max(a.workers, 1) / 60
    print(f"Estimativa (~2 s/requisição, {a.workers} threads): {est_min:.0f} min")
    if a.dry_run:
        return 0

    t0, done, failed = time.time(), 0, []
    if not a.merge_only and todo:
        if a.max_windows:
            todo = todo[: a.max_windows]
        ex = ThreadPoolExecutor(max_workers=a.workers)
        try:
            futs = [ex.submit(run_window, w, key, a.bbox, a.cell, outdir) for w in todo]
            for f in as_completed(futs):
                status, w, nraw, nagg = f.result()
                done += 1
                if status == "failed":
                    failed.append(w)
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    eta = el / done * (len(todo) - done)
                    print(f"  {done}/{len(todo)} janelas | falhas: {len(failed)} | ETA {eta/60:.1f} min", flush=True)
        except FatalAPIError as e:
            print(f"ERRO FATAL (chave inválida?): {e}")
            ex.shutdown(wait=False, cancel_futures=True)
            return 2
        except KeyboardInterrupt:
            print("\nInterrompido — salvando o que já foi baixado (rode de novo para retomar).")
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown()

    arch, meta = merge(outdir, windows, a.bbox, a.cell)
    if failed:
        (outdir / "failed.txt").write_text("\n".join(w.wid for w in failed))
        print(f"{len(failed)} janelas falharam (veja archive/failed.txt); rode de novo para tentar outra vez.")
    print(f"Arquivo: {outdir}/aeroguard_archive.{meta['format']} | {meta['rows']:,} linhas | "
          f"{meta['date_range']} | janelas faltando: {meta['windows_missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
