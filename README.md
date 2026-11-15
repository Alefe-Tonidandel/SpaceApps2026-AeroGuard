# AeroGuard — MODIS + VIIRS Hotspot Harmonization & Burning Activity Calendar

**🌐 Live Web Application (No Login Required):** [https://aeroguard-spaceapps.streamlit.app](https://aeroguard-spaceapps.streamlit.app)


NASA Space Apps Challenge 2026 · *Harmonization of MODIS and VIIRS Hot Spots*

Satellite active-fire records are split across sensors that cannot be compared directly (MODIS 1 km vs. VIIRS 375 m, different
overpasses and view geometry). AeroGuard harmonizes them into a **burning activity calendar** for any area of interest in Minas Gerais /
Cerrado transition (Brazil), 2010–2026, and turns it into answers for three audiences: **emergency responders, land managers, scientists**.

## What you can do in the app
| Tab | What it answers |
|---|---|
| Decision Brief | Persona switch (combat / prevention / scientist), plain-language or scientific. Live priority cells; fire-recurrence ("fire memory") zones |
| 1 Recent window | Live NRT detections → harmonized fire cells, calendar, diurnal matrix |
| 2 Cross-sensor validation | Near-simultaneous MODIS↔VIIRS pairs, FRP-density ratio with day-clustered CI, pixel-area dependence test |
| 3 2D DIP raster | Grid-aligned raster with separable Gaussian / box / LoG filters |
| 4 Methodology | Method, QA rules, limitations, harmonized-product export |
| 5 Historical calendar & alerts | Year × day-of-year calendar, P10–P90 band, anomaly percentile, era transition MODIS→VIIRS, custom AOI |

## Quick start
```bash
pip install -r requirements.txt
mkdir -p .streamlit && echo 'FIRMS_KEY = "YOUR_KEY"' > .streamlit/secrets.toml   # free key: firms.modaps.eosdis.nasa.gov/api/map_key
python3 scripts/build_archive.py --dry-run          # plan (reads FIRMS data availability live)
python3 scripts/build_archive.py                    # ~35 min, resumable; writes archive/aeroguard_archive.parquet
python3 scripts/coleta.py                           # recent NRT window (10 days) for tabs 1–4
streamlit run app.py
```
Optional: `NVIDIA_API_KEY` in secrets enables an AI rewrite of the plain-language bulletin (it is rejected and replaced by the deterministic text whenever it cites a number that is not in the data; other claims are not machine-checked). Never commit keys.

## Method in four steps
1. **Real pixel footprint.** FRP (MW) is extensive: it grows with the pixel area `scan × track` (MODIS median 1.2 km², up to 7.6 km² at scan edge;
   VIIRS median 0.23 km²). We derive FRP density (MW/km²) and aggregate to a grid; cell power is the **max** across sensors (they see the same fire).
2. **Pairing and validation.** Pair MODIS and VIIRS only within the same cell and |Δt| ≤ 90 min, one-to-one nearest match; report the ratio with a
   day-clustered bootstrap CI (pairs on the same day are not independent).
3. **Era harmonization.** Count *active 0.1° cells per day* (not pixels). Map each VIIRS platform to the MODIS scale on overlap days
   (ratio calibration), average the available estimates, validate with leave-one-year-out.
4. **Climatology and alerts.** Percentile of the current 7-day mean within the ±3-day window of the same day-of-year in all *other* years
   (P10–P90 band). Days with no data are gaps, never zeros.

## Results (Minas Gerais + Cerrado transition, W-51 S-23 E-39 N-14)
**Archive:** 2010-01-01 → 2026-11-14, 3,393,839 filtered detections → 1,415,265 (date × 0.1° cell × sensor) rows; 2,945 API windows, 0 failures.

**Cross-sensor pairs** (Sep 16–Oct 10 2026, 0.05° grid, |Δt| ≤ 90 min, n = 296):
- Median MODIS/VIIRS FRP-density ratio **0.31** (95% CI by day 0.25–0.41): VIIRS is systematically ~3× denser.
- The raw FRP ratio grows ~8× from the smallest to the largest area-ratio quartile (0.50 → 4.00); the density ratio varies ~1.7× (0.40 → 0.24).
  Even at near-equal pixel areas (Q1, area ratio 1.3×) VIIRS density is ~2.5× MODIS: the offset is **not** a pixel-size artifact.
- FRP scales sub-linearly with pixel area: slope 0.79 (95% CI 0.67–0.91), so area normalization removes most, not all, of the size effect.

**Era harmonization** (active cells/day, ratio mode):
| | k (VIIRS→MODIS scale) | Leave-one-year-out corr | Median \|annual bias\| |
|---|---|---|---|
| VIIRS SNPP | 0.377 | 0.83 | 3.5 % |
| VIIRS NOAA-20 | 0.371 | 0.86 | 4.5 % |

| Transition | Raw year-over-year | Harmonized | MODIS-only (control) | Natural year-to-year range (harmonized) |
|---|---|---|---|---|
| 2012 (SNPP enters) | ×4.43 | ×0.93 | ×0.93 | 0.69× – 1.60× |
| 2018 (NOAA-20 enters) | ×0.96 | ×0.69 | ×0.65 | 0.69× – 1.60× |

- **Deseasonalized anomaly correlation** (subtracting day-of-year median on log1p scale): **r = 0.621** (SNPP vs. MODIS) and **r = 0.699** (NOAA-20 vs. MODIS) for active cells/day (**0.606** and **0.708** for daily FRP).
- **2018 transition control:** The homogeneous MODIS-only series drops **×0.65** in 2018 (harmonized: **×0.69**; FRP: MODIS-only **×0.49** vs. harmonized **×0.53**). Because the harmonized series is anchored to MODIS, this agreement confirms that harmonization introduces no artificial step at NOAA-20's entry and is consistent with lower fire activity in 2018 (which the raw multi-sensor sum, **×0.96**, masked).
- **Fire-memory screening (Aug–Oct, 2010–2026):** Across 17 complete MODIS years, **276 priority 0.1° cells** (≈ 32,830 km², active in ≥90% of years and in the top 10% for active days/year; 2606 cells at the broader ≥70% tier out of 9929 cells with detections) stand out for pre-season prevention (855 priority cells across 15 complete VIIRS SNPP years, 2012–2026), with top Cerrado-transition hotspots at `(-15.15, -48.05)` and `(-15.05, -48.05)` active in 17/17 MODIS years and 15/15 SNPP years.

## Limitations (we state them on purpose)
- One region and one dataset family (FIRMS); no burned-area or fuel/topography layers yet (roadmap).
- Pairs are conditional on both sensors detecting the fire; same-day pairs are correlated (hence day-clustered CIs).
- Ratio calibration assumes the VIIRS↔MODIS relationship is stable over time and activity level (tested by leave-one-year-out, not proven).
- Hotspots include agricultural burns and are heat detections, not confirmed fire fronts or burned area.
- Edge-of-scan pixels (MODIS > 3 km²) locate fires less precisely and are flagged; flagging reduces, not removes, false alarms.
- VIIRS SNPP NRT stopped updating on 2026-11-02 (FIRMS data availability); calendar dates are UTC.
- Fire recurrence is a screening layer (cell is active if ≥1 detection), not a risk model.

## Data and references
- NASA FIRMS (MODIS C6.1, VIIRS SNPP/NOAA-20; standard-processing and NRT), via the FIRMS area API.
- Ribeiro, T.M., de Oliveira, J.F., Abreu, M.C. *Satellite-based assessment of fire hotspots in the Brazilian Amazon: implications of sensor differences
  and climatic drivers.* Theor Appl Climatol 157, 775 (2026). doi:10.1007/s00704-026-06701-1
