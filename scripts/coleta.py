import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import os
import pandas as pd
import firms_api

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None

MAP_KEY = os.environ.get("FIRMS_KEY", "").strip()
if not MAP_KEY and os.path.exists(".streamlit/secrets.toml"):
    if tomllib:
        with open(".streamlit/secrets.toml", "rb") as f:
            MAP_KEY = tomllib.load(f).get("FIRMS_KEY", "").strip()
    else:
        # Fallback para Python 3.10
        with open(".streamlit/secrets.toml", "r", encoding="utf-8") as f:
            for line in f:
                if "FIRMS_KEY" in line and "=" in line:
                    MAP_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")

AREA = "-51,-23,-39,-14"   # Minas Gerais + transição Cerrado
DIAS = 10                  # a API limita 5 dias por chamada; firms_api divide em janelas ('2026-11-05'+5d e '2026-11-10'+5d)

COLUNAS = [
    "data_hora", "data_hora_brt", "acq_date", "acq_date_brt", "acq_time", "daynight",
    "latitude", "longitude", "scan", "track", "bright_temp_k", "frp",
    "confidence", "satellite", "instrument", "sensor_source",
]

if __name__ == "__main__":
    if not MAP_KEY:
        raise SystemExit("Defina FIRMS_KEY (variável de ambiente ou .streamlit/secrets.toml).")

    df, logs, all_ok = firms_api.fetch_all(MAP_KEY, AREA, DIAS)
    print("\n".join(logs))
    if df.empty:
        raise SystemExit("Nenhum sensor retornou dados.")

    df["data_hora"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", errors="coerce")
    df["data_hora_brt"] = df["data_hora"] - pd.Timedelta(hours=3)
    df["acq_date_brt"] = df["data_hora_brt"].dt.strftime("%Y-%m-%d")
    df = df[[c for c in COLUNAS if c in df.columns]].sort_values("data_hora")

    print(f"\nPeríodo: {df.acq_date.min()} → {df.acq_date.max()} | sensores: {sorted(df.sensor_source.unique())}")
    if not all_ok:
        print("Aviso: algumas janelas/sensores falharam; arquivo salvo como PARCIAL.")
    df.to_csv("data/dados_incendios_mg.csv" if all_ok else "dados_incendios_mg_parcial.csv", index=False)
    print(f"Salvo: {len(df)} focos.")