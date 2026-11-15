"""
product_view.py — Aba "Decision Brief": a mesma base de dados, três públicos (Bilíngue PT/EN + Anti-removeChild).
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

import climatology as C

ARCHIVE = "archive/aeroguard_archive.parquet"
PRESETS = {
    "Entire archive area (MG + Cerrado transition)": (-51.0, -23.0, -39.0, -14.0),
    "Triângulo Mineiro / Alto Paranaíba (approx.)": (-51.0, -20.5, -47.0, -17.5),
    "North of Minas Gerais (approx.)": (-46.0, -18.0, -41.0, -14.0),
    "South of Minas / Mantiqueira (approx.)": (-47.0, -23.0, -42.0, -20.0),
}
MONTHS_PT = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun", 7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
EDGE_KM2 = 3.0
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.1-8b-instruct"


def complete_years(coverage, source, months, arch):
    end = arch["date"].max()
    if coverage is not None and len(coverage):
        end = max(end, coverage["end"].max())
    idx = pd.date_range(arch["date"].min(), end, freq="D")
    covd = C.coverage_days(coverage, [source], idx, arch)[source]
    ys = []
    for y in sorted(set(idx.year)):
        m = covd[(covd.index.year == y) & (covd.index.month.isin(months))]
        if len(m) and bool(m.all()):
            ys.append(y)
    return ys


def recurrence_table(arch, coverage, bbox, cell=0.1, months=(8, 9, 10), source="MODIS", recent_n=3):
    a = C.filter_aoi(arch, bbox, cell)
    a = a[(a["source"] == source) & a["date"].dt.month.isin(months)]
    yrs = complete_years(coverage, source, months, arch)
    meta = {"source": source, "months": list(months), "years": yrs, "n_years": len(yrs)}
    a = a[a["date"].dt.year.isin(yrs)]
    if a.empty or not yrs:
        return pd.DataFrame(), meta
    a = a.assign(year=a["date"].dt.year)
    g = a.groupby(["ix", "iy"]).agg(years_active=("year", "nunique"), active_days=("date", "nunique"),
                                    frp_total=("frp_sum", "sum"), last_year=("year", "max"))
    recent = yrs[-recent_n:]
    g["recent_active"] = a[a["year"].isin(recent)].groupby(["ix", "iy"])["year"].nunique().reindex(g.index).fillna(0).astype(int)
    g = g.reset_index()
    g["recurrence"] = g["years_active"] / len(yrs)
    g["days_per_active_year"] = g["active_days"] / g["years_active"]
    g["frp_per_active_year"] = g["frp_total"] / g["years_active"]
    g["lat"] = (g["iy"] + 0.5) * cell
    g["lon"] = (g["ix"] + 0.5) * cell
    g["cell_km2"] = (cell * 111.32) ** 2 * np.cos(np.radians(g["lat"]))
    dq = float(g["days_per_active_year"].quantile(0.9))
    g["priority"] = (g["recurrence"] >= 0.9) & (g["days_per_active_year"] >= dq)
    g["tier"] = np.where(g["priority"], "Priority (≥90% of years, top-10% active days)",
                         np.where(g["recurrence"] >= 0.7, "Chronic (≥70% of years)",
                                  np.where(g["recurrence"] >= 0.4, "Frequent (40–70%)", "Occasional (<40%)")))
    g = g.sort_values(["recurrence", "days_per_active_year", "frp_per_active_year"], ascending=False).reset_index(drop=True)
    meta["recent_years"] = recent
    meta["days_p90"] = dq
    return g, meta


def prevention_facts(rec, meta, aoi_name, top_n=3):
    chronic = rec[rec["recurrence"] >= 0.7]
    top = rec.head(top_n)
    return {
        "persona": "prevention", "aoi": aoi_name, "chronic_threshold_pct": 70,
        "months": "–".join([MONTHS[min(meta["months"])], MONTHS[max(meta["months"])]]),
        "month_nums": [int(min(meta["months"])), int(max(meta["months"]))],
        "sensor_basis": meta["source"], "n_years": meta["n_years"],
        "year_first": meta["years"][0], "year_last": meta["years"][-1],
        "n_cells": int(len(rec)), "n_chronic": int(len(chronic)),
        "chronic_area_km2": int(round(chronic["cell_km2"].sum(), -1)),
        "priority_threshold_pct": 90, "top_share_pct": 10,
        "n_priority": int(rec["priority"].sum()),
        "priority_area_km2": int(round(rec.loc[rec["priority"], "cell_km2"].sum(), -1)),
        "top": [{"lat": round(float(r.lat), 2), "lon": round(float(r.lon), 2),
                 "years_active": int(r.years_active), "days_per_year": round(float(r.days_per_active_year), 1)}
                for r in top.itertuples()],
    }


def live_priority(cells, window_days=2, n=10):
    if cells is None or cells.empty:
        return pd.DataFrame(), {}
    days = sorted(cells["acq_date"].astype(str).unique())[-window_days:]
    d = cells[cells["acq_date"].astype(str).isin(days)].copy()
    d["confirmed"] = d["n_families"] > 1
    d["edge_footprint"] = d["pixel_area_max"] > EDGE_KM2
    d["confidence"] = np.where(d["confirmed"], "Confirmed (MODIS + VIIRS)",
                               np.where(d["edge_footprint"], "Single sensor, large footprint — verify", "Single sensor"))
    d = d.sort_values(["frp_harm_mw"], ascending=False)
    top = d.head(n)
    facts = {
        "persona": "combat", "date_first": days[0], "date_last": days[-1], "window_days": len(days),
        "n_cells": int(len(d)), "n_confirmed": int(d["confirmed"].sum()), "n_edge": int(d["edge_footprint"].sum()),
        "top": [{"lat": round(float(r.grid_lat), 2), "lon": round(float(r.grid_lon), 2),
                 "frp_mw": round(float(r.frp_harm_mw), 0), "confirmed": bool(r.confirmed)} for r in top.head(3).itertuples()],
    }
    return top, facts


def template_bulletin(f, lang="en"):
    pt = lang == "pt"
    if f["persona"] == "combat":
        spots = "; ".join(f"({t['lat']}, {t['lon']}) ≈ {t['frp_mw']:.0f} MW" for t in f["top"])
        period = f["date_last"] if f["window_days"] == 1 else f"{f['date_first']} → {f['date_last']}"
        if pt:
            s = (f"Período {period}: {f['n_cells']} células com calor ativo detectado; {f['n_confirmed']} confirmadas por MODIS e VIIRS. "
                 f"Maiores potências: {spots}. ")
            if f["n_edge"]:
                s += f"{f['n_edge']} detecções têm pixel grande (borda da varredura): localização menos precisa, confirme antes de despachar. "
            return s + "Detecção de satélite indica calor, não confirma frente de fogo."
        s = (f"Period {period}: {f['n_cells']} cells with active heat detected; {f['n_confirmed']} confirmed by both MODIS and VIIRS. "
             f"Highest power: {spots}. ")
        if f["n_edge"]:
            s += f"{f['n_edge']} detections have a large satellite footprint (edge of scan): location is less precise, verify before dispatching. "
        return s + "A satellite detection indicates heat; it does not confirm a fire front."
    if f.get("n_priority", 0) > 0:
        if pt:
            top = "; ".join(f"({t['lat']}, {t['lon']}): fogo em {t['years_active']} de {f['n_years']} anos" for t in f["top"])
            return (f"Em {f['aoi']}, {f['n_priority']} células prioritárias (≈ {f['priority_area_km2']} km²) tiveram detecção de fogo em pelo menos 90% dos "
                    f"{f['n_years']} anos completos ({f['year_first']}–{f['year_last']}) e estão entre as 10% com mais dias ativos por ano, no período "
                    f"{'–'.join(MONTHS_PT[m] for m in f['month_nums'])}. Zonas mais recorrentes: {top}. Priorize aceiros e posicionamento de equipes nessas zonas "
                    "antes da temporada. É um mapa de triagem baseado só em detecções de satélite; combine com combustível, declividade e acesso.")
        top = "; ".join(f"({t['lat']}, {t['lon']}): fire in {t['years_active']} of {f['n_years']} years" for t in f["top"])
        return (f"In {f['aoi']}, {f['n_priority']} priority cells (≈ {f['priority_area_km2']} km²) had fire detections in at least 90% of the "
                f"{f['n_years']} complete years ({f['year_first']}–{f['year_last']}) and rank in the top 10% for active days per year during {f['months']}. "
                f"Most recurrent zones: {top}. Prioritize firebreaks and pre-positioning in these zones before the season. This is a screening map based only on "
                "satellite detections; combine it with fuel, slope and access data.")
    if pt:
        top = "; ".join(f"({t['lat']}, {t['lon']}): fogo em {t['years_active']} de {f['n_years']} anos" for t in f["top"])
        months_pt = "–".join(MONTHS_PT[m] for m in f["month_nums"])
        return (f"Em {f['aoi']}, {f['n_chronic']} células (≈ {f['chronic_area_km2']} km²) tiveram detecção de fogo em pelo menos 70% dos "
                f"{f['n_years']} anos completos ({f['year_first']}–{f['year_last']}) no período {months_pt}. Zonas mais recorrentes: {top}. "
                "Priorize aceiros e posicionamento de equipes nessas zonas antes da temporada. É um mapa de triagem baseado só em detecções de satélite; "
                "combine com combustível, declividade e acesso.")
    top = "; ".join(f"({t['lat']}, {t['lon']}): fire in {t['years_active']} of {f['n_years']} years" for t in f["top"])
    return (f"In {f['aoi']}, {f['n_chronic']} cells (≈ {f['chronic_area_km2']} km²) had fire detections in at least 70% of the "
            f"{f['n_years']} complete years ({f['year_first']}–{f['year_last']}) during {f['months']}. Most recurrent zones: {top}. "
            "Prioritize firebreaks and pre-positioning in these zones before the season. This is a screening map based only on satellite detections; "
            "combine it with fuel, slope and access data.")


def _num_tokens(text):
    return {"%g" % float(t.replace(",", ".")) for t in re.findall(r"\d+(?:[.,]\d+)?", text)}


def _fact_numbers(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _fact_numbers(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        v = float(obj)
        for fmt in ("{:.0f}", "{:.1f}", "{:.2f}"):
            out |= _num_tokens(fmt.format(v))
    elif isinstance(obj, str):
        out |= _num_tokens(obj)
    return out


def numbers_ok(text, facts):
    return _num_tokens(text) <= _fact_numbers(facts)


def llm_bulletin(facts, lang, api_key, model=NIM_MODEL, timeout=8):
    try:
        import requests
        persona = {"combat": "a wildfire brigade commander who has 20 seconds",
                   "prevention": "a land manager planning firebreaks before the fire season"}.get(facts["persona"], "a non-expert")
        sys_msg = ("You write short decision bulletins from satellite fire-detection facts. Use ONLY the JSON facts given. "
                   "Never invent places, causes, numbers, or fire sizes. Satellite detections show heat, not confirmed fire fronts: say so. "
                   f"Plain language, max 110 words, for {persona}. Write in {'Brazilian Portuguese' if lang == 'pt' else 'English'}.")
        r = requests.post(NIM_URL, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                          json={"model": model, "temperature": 0.2, "max_tokens": 260,
                                "messages": [{"role": "system", "content": sys_msg},
                                             {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]})
        if r.status_code != 200:
            return None
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text if text and numbers_ok(text, facts) else None
    except Exception:
        return None


def make_bulletin(facts, lang="en", use_ai=False, api_key=""):
    if use_ai and api_key:
        t = llm_bulletin(facts, lang, api_key)
        if t:
            return t, "ai"
    return template_bulletin(facts, lang), "template"


def _nim_key(st):
    k = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not k:
        try:
            k = str(st.secrets.get("NVIDIA_API_KEY", "")).strip()
        except Exception:
            k = ""
    return k


def render(st, cells, archive_path=ARCHIVE):
    c1, c2, c3 = st.columns([3, 1.6, 1])
    lang = c3.radio("Idioma / Language", ["PT", "EN"], index=0, horizontal=True, key="pv_lang").lower()
    pt = lang == "pt"

    st.subheader("🧭 Painel de Decisão — uma base de dados, três públicos" if pt else "🧭 Decision Brief — one dataset, three audiences")

    persona_opts = (
        {"🚒 Brigadista / Defesa Civil (Combate)": "combat",
         "🌱 Gestor Territorial / Prevenção": "prevention",
         "🔬 Cientista": "science"}
        if pt else
        {"🚒 Fire brigade / emergency responder (combat)": "combat",
         "🌱 Land manager / prevention planner": "prevention",
         "🔬 Scientist": "science"}
    )
    detail_opts = ["Linguagem simples", "Científico"] if pt else ["Plain language", "Scientific"]

    persona_lbl = c1.radio("Perfil de uso:" if pt else "I am a…", list(persona_opts), horizontal=True, key="pv_persona")
    persona = persona_opts[persona_lbl]
    detail_lbl = c2.radio("Nível de detalhe:" if pt else "Detail level", detail_opts, horizontal=True, key="pv_detail")
    sci = (detail_lbl in ("Científico", "Scientific")) or persona == "science"

    key = _nim_key(st)
    use_ai = False
    if persona != "science" and key:
        use_ai = st.checkbox(
            "✨ Reescrever resumo com IA (NVIDIA NIM)" if pt else "✨ Rewrite the summary with AI (NVIDIA NIM)",
            value=False, key="pv_ai"
        )

    def show_bulletin(facts):
        text, origin = make_bulletin(facts, lang, use_ai, key)
        st.info(text)
        if pt:
            st.caption("Gerado por IA e validado contra os dados acima." if origin == "ai"
                       else "Resumo operacional automático gerado diretamente a partir dos números verificados acima.")
        else:
            st.caption("Written by AI from the numbers above; numbers were checked against the data." if origin == "ai"
                       else "Automatic summary generated from the numbers above.")

    slot_combat = st.container()
    slot_prev = st.container()
    slot_sci = st.container()

    if persona == "combat":
        with slot_combat:
            st.markdown("#### Onde está queimando agora? *(janela NRT ao vivo, MODIS + VIIRS harmonizados)*" if pt
                        else "#### Where is it burning right now? *(live NRT window, MODIS + VIIRS harmonized)*")
            top, f = live_priority(cells, window_days=2, n=10)
            if not f:
                st.warning("Sem detecções recentes na janela carregada." if pt else "No recent detections in the loaded window.")
                return
            m1, m2, m3 = st.columns(3)
            m1.metric("Células de fogo ativas" if pt else "Active fire cells", f["n_cells"],
                      f"últimos {f['window_days']} dia(s)" if pt else f"last {f['window_days']} day(s) of data")
            m2.metric("Confirmadas por 2 famílias (MODIS+VIIRS)" if pt else "Confirmed by 2 sensor families", f["n_confirmed"])
            m3.metric("Pegada grande na borda (verificar)" if pt else "Large footprint (verify)", f["n_edge"],
                      help="Pixel MODIS > 3 km² (borda da varredura): localização menos precisa" if pt
                           else "MODIS pixel > 3 km²: edge of the scan, location less precise")
            show_bulletin(f)
            t = top.assign(Latitude=top["grid_lat"].round(3), Longitude=top["grid_lon"].round(3),
                           **{"Fire power (MW)": top["frp_harm_mw"].round(0), "Confidence": top["confidence"]})
            cols = ["Latitude", "Longitude", "Fire power (MW)", "Confidence"]
            if sci:
                t = t.assign(**{"FRP density (MW/km²)": top["frpd_harm"].round(1), "Max pixel (km²)": top["pixel_area_max"].round(2),
                                "Sensors": top["sensors"], "Date": top["acq_date"]})
                cols += ["FRP density (MW/km²)", "Max pixel (km²)", "Sensors", "Date"]
            st.dataframe(t[cols].head(10 if sci else 5), hide_index=True)
            st.map(top.rename(columns={"grid_lat": "latitude", "grid_lon": "longitude"})[["latitude", "longitude"]].head(10), size=300)
            st.caption("Confirmação dupla de sensores e tamanho da pegada reduzem falsos alarmes, mas não os eliminam. Detecções indicam calor, não frentes confirmadas." if pt
                       else "Dual-sensor confirmation and footprint size reduce false alarms but do not eliminate them. Satellite detections show heat, not confirmed fire fronts.")

    elif persona == "prevention":
        with slot_prev:
            if not Path(archive_path).exists():
                st.warning(f"Arquivo histórico não encontrado (`{archive_path}`). Rode `python3 build_archive.py`." if pt
                           else f"Historical archive not found (`{archive_path}`). Run `python3 build_archive.py`.")
                return

            @st.cache_resource(show_spinner="Loading historical archive...")
            def _load(path, stamp):
                return C.load_archive(path)

            @st.cache_data(show_spinner="Computing fire recurrence...")
            def _rec(path, stamp, bbox, months, source):
                arch, meta, cov = _load(path, stamp)
                return recurrence_table(arch, cov, bbox, meta.get("cell_deg", 0.1), months, source)

            st.markdown("#### Onde o fogo volta todo ano? *(memória de fogo, histórico 2010–2026)*" if pt
                        else "#### Where does fire come back every year? *(fire memory, archive 2010–2026)*")
            p1, p2, p3 = st.columns([2, 2, 2])
            aoi = p1.selectbox("Área de análise" if pt else "Area", list(PRESETS), key="pv_aoi")
            months = p2.multiselect("Meses da temporada" if pt else "Fire season months", list(MONTHS), default=[8, 9, 10],
                                    format_func=lambda m: MONTHS_PT[m] if pt else MONTHS[m], key="pv_months") or [8, 9, 10]
            basis = p3.selectbox("Base de sensores" if pt else "Sensor basis",
                                 ["MODIS only (2010–2026, homogeneous)", "VIIRS Suomi-NPP only (2012–2026, more sensitive)"], key="pv_basis")
            source = "MODIS" if basis.startswith("MODIS") else "SNPP"
            rec, meta = _rec(archive_path, os.path.getmtime(archive_path), tuple(PRESETS[aoi]), tuple(sorted(months)), source)
            if rec.empty:
                st.warning("Nenhum ano completo com detecções para esta seleção." if pt else "No complete years with detections for this selection.")
                return
            f = prevention_facts(rec, meta, aoi)
            m1, m2, m3 = st.columns(3)
            m1.metric("Anos completos analisados" if pt else "Complete years analysed", meta["n_years"], f"{meta['years'][0]}–{meta['years'][-1]}")
            m2.metric("Zonas prioritárias (≥90% + top 10% dias)" if pt else "Priority zones", f["n_priority"], f"≈ {f['priority_area_km2']:,} km²",
                      help="Fogo em ≥90% dos anos E entre os 10% com mais dias ativos/ano." if pt
                           else "Fire in ≥90% of the years AND in the top 10% of cells for active days per year. The strictest tier.")
            m3.metric("Células com fogo em ≥70% dos anos" if pt else "Cells with fire in ≥70% of years", f["n_chronic"],
                      help="Célula de 0,1° (~120 km²) conta como ativa se teve ≥1 detecção no período." if pt
                           else "A 0.1° cell (~120 km²) counts if at least one detection fell in it, so this tier is broad.")
            show_bulletin(f)
            n = st.slider("Zonas a listar" if pt else "Zones to list", 10, 50, 20, step=5, key="pv_slider")
            top = rec.head(n)
            t = top.assign(Latitude=top["lat"].round(3), Longitude=top["lon"].round(3),
                           **{"Years with fire": top["years_active"].astype(str) + f" of {meta['n_years']}",
                              "Avg. active days / year": top["days_per_active_year"].round(1), "Tier": top["tier"]})
            cols = ["Latitude", "Longitude", "Years with fire", "Avg. active days / year", "Tier"]
            if sci:
                t = t.assign(**{"FRP / active year (MW)": top["frp_per_active_year"].round(0),
                                f"Active in last {len(meta['recent_years'])} yrs": top["recent_active"], "Recurrence": top["recurrence"].round(2)})
                cols += ["FRP / active year (MW)", f"Active in last {len(meta['recent_years'])} yrs", "Recurrence"]
            st.dataframe(t[cols], hide_index=True)
            st.map(top.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]], size=600)
            st.download_button("⬇️ Baixar todas as zonas (CSV)" if pt else "⬇️ Download all zones (CSV)",
                               rec.to_csv(index=False).encode(), file_name="aeroguard_fire_recurrence.csv", mime="text/csv")
            st.caption("Camada de triagem baseada em detecções de satélite (inclui queima agrícola). Combine com combustível, declividade e acesso (roadmap) antes de planejar aceiros." if pt
                       else "Screening layer from satellite detections (agricultural burns included). Not a risk model: add fuel, slope and access data (roadmap) before planning firebreaks.")

    else:
        with slot_sci:
            st.markdown("#### Validação científica e metodologia" if pt else "#### Validation and method")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pares simultâneos MODIS↔VIIRS" if pt else "Simultaneous MODIS↔VIIRS pairs", "296", "|Δt| ≤ 90 min (0.05°)")
            m2.metric("Razão mediana de densidade FRP" if pt else "Median FRP-density ratio", "0.31×", "95% CI: 0.25–0.41")
            m3.metric("Série histórica harmonizada" if pt else "Harmonized historical series", "17 anos" if pt else "17 years", "2010–2026 (0 falhas)")
            if pt:
                st.info("Validação cruzada entre eras (MODIS ↔ VIIRS): k = 0,377 (SNPP, viés anual mediano LOYO de 3,5%) e k = 0,371 (NOAA-20, viés de 4,5%). O salto bruto de 2012 (×4,43) é corrigido para ×0,93 (controle MODIS-only: ×0,93).")
                st.caption("Resumo metodológico verificável nas Abas 2, 4 e 5.")
                st.markdown("""
- **Aba 2** — pares quase simultâneos MODIS↔VIIRS: razão de densidade FRP (`MW/km²`) com IC de 95% agrupado por dia e teste de dependência da área do pixel.
- **Aba 4** — metodologia física, regras de controle de qualidade (QA), limitações e exportação do produto harmonizado.
- **Aba 5** — harmonização entre eras (MODIS → VIIRS, 2010–2026), validação *leave-one-year-out*, tabela de transição com controle *MODIS-only*, faixa histórica P10–P90 e anomalias.
- Todos os números deste Painel de Decisão vêm diretamente dessas tabelas verificadas.
""")
            else:
                st.info("Cross-era validation (MODIS ↔ VIIRS): k = 0.377 (SNPP, LOYO median annual bias 3.5%) and k = 0.371 (NOAA-20, bias 4.5%). The 2012 raw jump (×4.43) is corrected to ×0.93 (MODIS-only control: ×0.93).")
                st.caption("Methodological summary verifiable in Tabs 2, 4, and 5.")
                st.markdown("""
- **Tab 2** — near-simultaneous MODIS↔VIIRS pairs: FRP-density ratio with day-clustered 95% CI, and the pixel-area dependence test.
- **Tab 4** — method, QA rules and limitations; export of the harmonized product.
- **Tab 5** — era harmonization (MODIS → VIIRS), leave-one-year-out check, year-over-year step table, historical P10–P90 and anomalies.
- Every number in the Decision Brief comes from these same tables; the AI rewrite (if enabled) is rejected whenever it cites a number not in the data.
""")
