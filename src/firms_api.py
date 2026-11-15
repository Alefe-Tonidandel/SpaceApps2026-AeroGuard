"""firms_api.py — cliente NASA FIRMS compartilhado por coleta.py e app.py.

A API de área aceita DAY_RANGE de 1 a 5. Para janelas maiores, fazemos várias
requisições de até 5 dias usando o parâmetro opcional DATE (início da janela).
"""
import datetime as dt
import io
import time

import pandas as pd
import requests

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_RANGE = 5
SENSORES = ["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]


def _get(url: str, retries: int = 3, timeout: int = 35):
    """Retorna (df | None, mensagem). Erros de cliente (chave, range, limite) NÃO são re-tentados."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            txt = r.text.strip()
            if r.status_code == 200 and txt.startswith("latitude"):
                return pd.read_csv(io.StringIO(txt)), "ok"
            if r.status_code == 200 and not txt:
                return pd.DataFrame(), "empty"
            low = txt.lower()
            if r.status_code < 500 or "invalid" in low or "exceed" in low:
                return None, txt[:120] or f"HTTP {r.status_code}"   # erro determinístico: não insistir
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * attempt)
    return None, last


def fetch_sensor(sensor: str, key: str, area: str, days: int, end_date: dt.date | None = None):
    """Baixa `days` dias de um sensor, em janelas de até 5 dias. Retorna (df, logs, ok)."""
    logs, parts, ok = [], [], True
    if days <= MAX_RANGE and end_date is None:           # caso simples: "mais recentes", sem DATE
        windows = [(None, days)]
    else:
        end = end_date or dt.datetime.now(dt.timezone.utc).date()
        start = end - dt.timedelta(days=days - 1)
        windows, cur = [], start
        while cur <= end:
            n = min(MAX_RANGE, (end - cur).days + 1)
            windows.append((cur, n))
            cur += dt.timedelta(days=n)

    for start, n in windows:
        url = f"{BASE}/{key}/{sensor}/{area}/{n}" + (f"/{start.isoformat()}" if start else "")
        df, msg = _get(url)
        tag = f"{sensor} [{start or 'latest'}+{n}d]"
        if df is None:
            ok = False
            logs.append(f"{tag}: {msg}")
            if "invalid" in msg.lower() or "exceed" in msg.lower():
                break                                    # chave/limite: parar este sensor
        else:
            logs.append(f"{tag}: {len(df)} focos")
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(), logs, ok
    out = pd.concat(parts, ignore_index=True)
    out["sensor_source"] = sensor
    out = out.rename(columns={"brightness": "bright_temp_k", "bright_ti4": "bright_temp_k"})
    return out.drop_duplicates(), logs, ok


def fetch_all(key: str, area: str, days: int, sensores=SENSORES):
    """Retorna (df_all, logs, all_ok). all_ok=True só se TODAS as requisições deram certo."""
    frames, logs, all_ok = [], [], True
    for s in sensores:
        df, lg, ok = fetch_sensor(s, key, area, days)
        logs += lg
        all_ok &= ok
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), logs, False
    return pd.concat(frames, ignore_index=True), logs, all_ok
