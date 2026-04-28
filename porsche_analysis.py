import os
import webbrowser
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import process
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 1. SCRAPE NÜRBURGRING LAP TIMES ──────────────────────────────────────────

# scrape the lap time table from fastestlaps.com and convert MM:SS to total seconds
def get_lap_times():
    url = "https://fastestlaps.com/tracks/nordschleife"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        laps = []
        table = soup.find("table")
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) > 3:
                name = cols[1].text.strip()
                time_str = cols[3].text.strip()
                try:
                    # convert "7:32.1" → 452.1 seconds
                    m, s = time_str.split(":")
                    total_seconds = int(m) * 60 + float(s)
                    laps.append({"web_name": name, "lap_seconds": total_seconds})
                except:
                    continue
        return pd.DataFrame(laps)
    except Exception as e:
        print(f"  Warning: could not fetch lap times ({e})")
        return pd.DataFrame(columns=["web_name", "lap_seconds"])


# fuzzy match a car name against the scraped list — only accept if 80%+ similar
def find_best_match(name, choices):
    match, score = process.extractOne(name, choices)
    return match if score > 80 else None


# fetch the lap data once at startup
print("Fetching Nürburgring lap times …")
lap_df = get_lap_times()
print(f"  → {len(lap_df)} entries scraped")
# ── PHASE 2: DATA PREPARATION ──
# ── 2. LOAD & CLEAN ────────────────────────────────────────────────────────────

# load the Kaggle Porsche 911 dataset (~300 variants)
df = pd.read_csv("data/porsche_911.csv")
df.columns = df.columns.str.strip()


# assign each car to a generation based on its production start year
def categorize_generation(year):
    if year <= 1973:   return "Original"
    elif year <= 1989: return "G-Series"
    elif year <= 1994: return "964"
    elif year <= 1998: return "993"
    elif year <= 2004: return "996"
    elif year <= 2012: return "997"
    elif year <= 2019: return "991"
    else:              return "992"


df["start_of_production"] = pd.to_numeric(df["start_of_production"], errors="coerce")
df["generation"] = df["start_of_production"].apply(
    lambda y: categorize_generation(y) if pd.notna(y) else "Unknown"
)

# 996 onward (1999+) switched from air-cooled to water-cooled engines
df["cooling_type"] = df["start_of_production"].apply(
    lambda y: "Water-Cooled" if pd.notna(y) and y >= 1999 else "Air-Cooled"
)

# strip units and convert every column we need to plain numbers
df["power"]                = df["power"].str.extract(r"(\d+)").astype(float)
df["kerb_weight"]          = pd.to_numeric(df["kerb_weight"],          errors="coerce")
df["acceleration_0-60mph"] = pd.to_numeric(df["acceleration_0-60mph"], errors="coerce")
df["engine_displacement"]  = pd.to_numeric(df["engine_displacement"],  errors="coerce")
df["power_per_litre"]      = pd.to_numeric(df["power_per_litre"],      errors="coerce")
df["front_track"]          = pd.to_numeric(df["front_track"],          errors="coerce")
df["rear_track"]           = pd.to_numeric(df["rear_track"],           errors="coerce")

# drop rows that are missing the three metrics every chart depends on
df = df.dropna(subset=["power", "kerb_weight", "acceleration_0-60mph"])
df = df[df["acceleration_0-60mph"] > 0].copy()

# try to attach a Nürburgring lap time to each car using fuzzy name matching
if not lap_df.empty:
    lap_choices = lap_df["web_name"].tolist()
    df["matched_name"] = df["engine"].apply(
        lambda x: find_best_match(str(x), lap_choices)
    )
    # left join so cars without a match just get NaN for lap_seconds
    df = pd.merge(df, lap_df, left_on="matched_name", right_on="web_name", how="left")
    n_matched = df["lap_seconds"].notna().sum()
    print(f"  → {n_matched} cars matched to Nürburgring lap times")
else:
    df["lap_seconds"] = np.nan
    df["matched_name"] = np.nan

# ── PHASE 3: ANALYSIS & VISUALIZATION ──
# ── 3. DERIVED METRICS ─────────────────────────────────────────────────────────

# power-to-weight ratio — higher means more punch per kilogram
df["hp_per_kg"] = df["power"] / df["kerb_weight"]

# driver score using 0-60 time as a proxy — available for every car in the dataset
df["drivers_score"] = (df["hp_per_kg"] * 100) / df["acceleration_0-60mph"]
# normalize to 0-100 so the best car scores exactly 100
_min, _max = df["drivers_score"].min(), df["drivers_score"].max()
df["score_norm"] = ((df["drivers_score"] - _min) / (_max - _min)) * 100

# Nurburgring score using real lap seconds — only calculated where we have scraped data
df["nbr_score_raw"] = np.where(
    df["lap_seconds"].notna(),
    (df["hp_per_kg"] * 100) / df["lap_seconds"],
    np.nan,
)
# normalize the Nurburgring score the same way, 0-100
nbr_vals = df["nbr_score_raw"].dropna()
if len(nbr_vals) > 1:
    _nmin, _nmax = nbr_vals.min(), nbr_vals.max()
    df["nbr_score"] = ((df["nbr_score_raw"] - _nmin) / (_nmax - _nmin)) * 100
else:
    df["nbr_score"] = df["nbr_score_raw"]

# how much wider the rear track is vs the front — bigger = more planted rear end
df["track_delta"] = df["rear_track"] - df["front_track"]

# flag US-spec cars that had a catalytic converter fitted (caused power loss in the 70s-80s)
df["is_cat"] = df["engine"].str.contains("CAT", case=False, na=False)

# ── 4. CONSOLE STATS ──────────────────────────────────────────────────────────

# fixed order so generation tables always read oldest → newest
GEN_ORDER = ["Original", "G-Series", "964", "993", "996", "997", "991", "992"]

# print the top 10 cars ranked by driver score
print("=" * 60)
print("TOP 10 BY DRIVER SCORE")
print("=" * 60)
top10_cols = ["engine", "generation", "power", "kerb_weight",
              "acceleration_0-60mph", "score_norm", "cooling_type"]
print(
    df[top10_cols]
    .sort_values("score_norm", ascending=False)
    .head(10)
    .to_string(index=False)
)

# show average weight, power, and score for each generation
print("\n" + "=" * 60)
print("GENERATION AVERAGES")
print("=" * 60)
gen_stats = (
    df.groupby("generation")
    .agg(
        avg_weight=("kerb_weight", "mean"),
        avg_power=("power", "mean"),
        avg_score=("score_norm", "mean"),
        avg_hp_per_litre=("power_per_litre", "mean"),
        count=("power", "count"),
    )
    .reindex(GEN_ORDER)
    .round(1)
)
print(gen_stats.to_string())

# compare air-cooled vs water-cooled cars across key metrics
print("\n" + "=" * 60)
print("AIR VS WATER COOLED")
print("=" * 60)
cooling_stats = (
    df.groupby("cooling_type")
    .agg(
        avg_weight=("kerb_weight", "mean"),
        avg_power=("power", "mean"),
        avg_score=("score_norm", "mean"),
        avg_hp_per_kg=("hp_per_kg", "mean"),
        count=("power", "count"),
    )
    .round(3)
)
print(cooling_stats.to_string())

# calculate how much power US-spec CAT cars lost vs the non-CAT versions
print("\n" + "=" * 60)
print("70s–80s EMISSIONS PENALTY (CAT variants, G-Series)")
print("=" * 60)
g_series = df[df["generation"] == "G-Series"]
cat_avg   = g_series[g_series["is_cat"]]["power"].mean()
nocat_avg = g_series[~g_series["is_cat"]]["power"].mean()
print(f"  Non-CAT avg power : {nocat_avg:.0f} hp")
print(f"  CAT avg power     : {cat_avg:.0f} hp")
print(f"  Penalty           : {nocat_avg - cat_avg:.0f} hp ({(nocat_avg - cat_avg) / nocat_avg * 100:.1f}%)")


# ── 5. PORSCHE THEME ─────────────────────────────────────────────────────────

COLOR_MAP = {
    "Original": "#98c1d9",   # Powder Blue
    "G-Series": "#3d5a80",   # Dusk Blue
    "964":      "#7ab3d4",
    "993":      "#5b9ec9",
    "996":      "#ee6c4d",   # Burnt Peach
    "997":      "#f08d72",
    "991":      "#e0fbfc",   # Light Cyan
    "992":      "#b0e8ea",
}

DARK_BG    = "#293241"   # Jet Black
DARK_PAPER = "#1e2a38"   # surface
DARK_GRID  = "#2d3d52"   # subtle gridlines
DARK_TEXT  = "#e0fbfc"   # Light Cyan
TITLE_COL  = "#e0fbfc"
ACCENT     = "#ee6c4d"   # Burnt Peach

BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_PAPER,
    font=dict(color=DARK_TEXT, family="'Inter', 'Segoe UI', sans-serif", size=12),
    title_font=dict(size=15, color=TITLE_COL, family="'Inter', 'Segoe UI', sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=DARK_TEXT)),
)

AX = dict(
    gridcolor="#2d3d52",
    linecolor="#3d5a80",
    zerolinecolor="#3d5a80",
    tickfont=dict(color="#98c1d9"),
    title_font=dict(color="#98c1d9"),
)

# ── 6. CHARTS ─────────────────────────────────────────────────────────────────

# Chart 1: horizontal bar chart ranking the top 20 cars by driver score
top20 = df.nlargest(20, "score_norm")[
    ["engine", "generation", "power", "kerb_weight",
     "acceleration_0-60mph", "score_norm", "cooling_type"]
].copy()
top20["label"] = top20["engine"].str.strip().str[:40]

fig1 = px.bar(
    top20.sort_values("score_norm"),
    x="score_norm", y="label",
    color="generation",
    color_discrete_map=COLOR_MAP,
    orientation="h",
    title="Top 20 Porsche 911 variants — Driver Score",
    labels={"score_norm": "Driver Score (higher = better)", "label": ""},
    hover_data={"power": True, "kerb_weight": True,
                "acceleration_0-60mph": True, "generation": True},
)
fig1.update_layout(**BASE, legend_title="Generation", height=600,
                   margin=dict(l=260), xaxis=AX,
                   yaxis={**AX, "tickfont": dict(color=DARK_TEXT, size=10)})

# Chart 2: scatter plot showing whether lighter cars score better (sweet spot zone highlighted)
fig2 = px.scatter(
    df,
    x="kerb_weight", y="score_norm",
    color="generation",
    color_discrete_map=COLOR_MAP,
    symbol="cooling_type",
    hover_name="engine",
    hover_data={"power": True, "kerb_weight": True,
                "acceleration_0-60mph": True, "score_norm": ":.1f"},
    title="Sweet Spot — Weight vs Driver Score",
    labels={"kerb_weight": "Kerb weight (kg)", "score_norm": "Driver Score (0–100)"},
)
fig2.add_shape(
    type="rect", x0=1250, x1=1500, y0=35, y1=105,
    fillcolor="rgba(26,111,196,0.08)",
    line=dict(color="rgba(26,111,196,0.45)", width=1, dash="dot"),
)
fig2.add_annotation(
    x=1375, y=108, text="Sweet spot", showarrow=False,
    font=dict(size=11, color=ACCENT),
)
fig2.update_layout(**BASE, legend_title="Generation", height=520, xaxis=AX, yaxis=AX)

# Chart 3: line chart showing how engine efficiency (HP per litre) improved generation by generation
hp_litre = (
    df.dropna(subset=["power_per_litre"])
    .groupby("generation")["power_per_litre"]
    .mean()
    .reindex(GEN_ORDER)
    .reset_index()
)
hp_litre.columns = ["generation", "hp_per_litre"]

fig3 = px.line(
    hp_litre, x="generation", y="hp_per_litre",
    markers=True,
    title="Engine efficiency — avg HP per litre by generation",
    labels={"generation": "Generation", "hp_per_litre": "HP / litre"},
)
fig3.update_traces(line_color=ACCENT, marker_size=9,
                   marker_color="#98c1d9", line_width=2)
fig3.update_layout(**BASE, height=420, xaxis=AX, yaxis=AX)

# Chart 4: side-by-side bars comparing air-cooled and water-cooled across weight, power, and score
cooling_plot = cooling_stats.reset_index()
subplot_labels = ["Avg weight (kg)", "Avg power (hp)", "Avg driver score"]
fig4 = make_subplots(rows=1, cols=3, subplot_titles=subplot_labels)
colors_cool = {"Air-Cooled": "#98c1d9", "Water-Cooled": ACCENT}

for i, col in enumerate(["avg_weight", "avg_power", "avg_score"], start=1):
    for _, row in cooling_plot.iterrows():
        fig4.add_trace(
            go.Bar(
                name=row["cooling_type"],
                x=[row["cooling_type"]],
                y=[row[col]],
                marker_color=colors_cool[row["cooling_type"]],
                showlegend=(i == 1),
            ),
            row=1, col=i,
        )

fig4.update_layout(
    **BASE,
    title_text="Air-Cooled vs Water-Cooled — key averages",
    height=400,
    barmode="group",
    legend_title="Cooling type",
)
for ax_key in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
    fig4.update_layout(**{ax_key: AX})
for ann in fig4.layout.annotations:
    ann.update(font=dict(color=DARK_TEXT))

# Chart 5: strip chart showing each G-Series car's power output — CAT vs non-CAT side by side
g_detail = g_series[["engine", "power", "is_cat", "start_of_production"]].copy()
g_detail["variant"] = g_detail["is_cat"].map({True: "CAT (emissions)", False: "Non-CAT"})

fig5 = px.strip(
    g_detail,
    x="start_of_production", y="power",
    color="variant",
    hover_name="engine",
    title="G-Series (1974–1989) — emissions impact on power output",
    labels={"start_of_production": "Year", "power": "Power (hp)", "variant": "Variant type"},
    color_discrete_map={"CAT (emissions)": ACCENT, "Non-CAT": "#98c1d9"},
)
fig5.update_layout(**BASE, height=420, xaxis=AX, yaxis=AX)

# Chart 6: scatter with a smoothed trend line showing how power-to-weight has grown over the decades
fig6 = px.scatter(
    df,
    x="start_of_production", y="hp_per_kg",
    color="cooling_type",
    trendline="lowess",
    hover_name="engine",
    title="HP per kg over time — with smoothed trend",
    labels={"start_of_production": "Year", "hp_per_kg": "HP / kg", "cooling_type": "Cooling"},
    color_discrete_map={"Air-Cooled": "#98c1d9", "Water-Cooled": ACCENT},
)
fig6.update_layout(**BASE, height=460, xaxis=AX, yaxis=AX)

figs = [fig1, fig2, fig3, fig4, fig5, fig6]

# descriptions and anchor IDs for each chart — used to build the sidebar and chart headers
chart_meta = [
    {"id": "driver-score", "label": "Driver Score",
     "description": "Ranks every 911 variant using (HP/Weight) divided by 0-60 time, normalized so the best car scores exactly 100. Higher scores reward cars that deliver the most performance per kilogram as quickly as possible."},
    {"id": "sweet-spot", "label": "Weight vs Driver Score",
     "description": "Each dot is a single 911 variant plotted by kerb weight against its driver score. The shaded zone marks where Porsche consistently found the best balance between agility and outright performance."},
    {"id": "hp-per-litre", "label": "Engine Efficiency",
     "description": "Average horsepower extracted from every litre of displacement, grouped by generation. The flat middle period reflects the emissions era; the steep climb from the 997 onwards shows what direct injection and modern turbocharging unlocked."},
    {"id": "air-vs-water", "label": "Air vs Water-Cooled",
     "description": "Porsche switched to water cooling with the 996 in 1999, a move that divided fans but delivered a step-change in power and refinement. These three bars compare average weight, power output, and driver score across both eras."},
    {"id": "emissions", "label": "Emissions Penalty",
     "description": "The US Clean Air Act forced American-spec G-Series 911s to run catalytic converters, visibly cutting power output versus European variants. Each dot is a real car, showing the exact performance cost of emissions compliance year by year."},
    {"id": "hp-kg-trend", "label": "HP/kg Over Time",
     "description": "A full timeline of every 911 variant power-to-weight ratio with a LOWESS smoothed trend line. The plateau in the late 1970s marks the emissions era; the sharp climb post-2005 shows what modern forced induction achieved."},
]

# Chart 7: rank cars using the real lap-time formula — only shown if the scraper got matches
nbr_df = df.dropna(subset=["nbr_score"]).sort_values("nbr_score", ascending=False).head(20).copy()
if len(nbr_df) > 0:
    nbr_df["label"] = nbr_df["engine"].str.strip().str[:40]
    fig7 = px.bar(
        nbr_df.sort_values("nbr_score"),
        x="nbr_score", y="label",
        color="generation",
        color_discrete_map=COLOR_MAP,
        orientation="h",
        title="Nürburgring Score — (HP/kg) ÷ Lap Time (matched cars only)",
        labels={"nbr_score": "Nürburgring Score (higher = better)", "label": ""},
        hover_data={"lap_seconds": True, "power": True, "generation": True},
    )
    fig7.update_layout(
        **BASE,
        legend_title="Generation",
        height=max(400, len(nbr_df) * 28),
        margin=dict(l=280),
        xaxis=AX,
        yaxis={**AX, "tickfont": dict(color=DARK_TEXT, size=10)},
    )
    figs.append(fig7)
    chart_meta.append({
        "id": "nbr-score", "label": "Nurburgring Score",
        "description": "The original goal formula: (HP/kg) divided by actual Nordschleife lap time, normalized 0-100. Only cars with a matched lap time scraped from fastestlaps.com are included. Both a faster lap and a higher power-to-weight ratio push the score up.",
    })
else:
    print("  No Nürburgring matches — skipping chart 7")

# Chart 8: does a wider rear track actually help lap times? scatter + trend line to find out
track_lap = df.dropna(subset=["track_delta", "lap_seconds"]).copy()
if len(track_lap) >= 3:
    fig8 = px.scatter(
        track_lap,
        x="track_delta", y="lap_seconds",
        color="generation",
        color_discrete_map=COLOR_MAP,
        trendline="lowess",
        hover_name="engine",
        hover_data={"power": True, "track_delta": True, "lap_seconds": True},
        title="Does a wider rear track mean faster laps? Track delta vs Nürburgring time",
        labels={
            "track_delta": "Rear − front track width (mm)",
            "lap_seconds": "Lap time (seconds)",
        },
    )
    fig8.update_layout(**BASE, height=460, xaxis=AX, yaxis=AX)
    figs.append(fig8)
    chart_meta.append({
        "id": "track-delta", "label": "Track Width vs Lap Time",
        "description": "Tests whether a wider rear track relative to the front correlates with faster Nurburgring lap times. A downward slope would confirm the hypothesis. Only cars with both track measurement data and a matched lap time are included.",
    })
else:
    print("  Not enough track+lap overlap — skipping chart 8")

# ── 7. BUILD WEBPAGE ──────────────────────────────────────────────────────────

# compute headline stats shown in the sidebar stats panel
year_min       = int(df["start_of_production"].min())
year_max       = int(df["start_of_production"].max())
n_lap_matches  = int(df["lap_seconds"].notna().sum())

# build generation legend from COLOR_MAP (same order as charts)
legend_items_html = ""
for gen in GEN_ORDER:
    color = COLOR_MAP.get(gen, "#888")
    legend_items_html += (
        f'<div class="legend-item">'
        f'<span class="legend-dot" style="background:{color}"></span>{gen}'
        f'</div>'
    )

# build sidebar navigation — one link per chart, dot indicator via CSS ::before
nav_links_parts = []
for m in chart_meta:
    mid    = m["id"]
    mlabel = m["label"]
    nav_links_parts.append(f'<li><a href="#{mid}" class="nav-link">{mlabel}</a></li>')
nav_links = "\n        ".join(nav_links_parts)

# build each chart section: red-accented description header + plotly chart body
# only the first chart loads plotly.js (via CDN); the rest reference it
section_parts = []
for i, (fig, meta) in enumerate(zip(figs, chart_meta)):
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
    sec_id    = meta["id"]
    sec_label = meta["label"]
    sec_desc  = meta["description"]
    delay     = f"{i * 0.12:.2f}"
    section_parts.append(
        f'<section class="chart-section" id="{sec_id}" style="animation-delay:{delay}s">'
        f'<div class="chart-header"><h2>{sec_label}</h2>'
        f'<p class="chart-desc">{sec_desc}</p></div>'
        f'<div class="chart-body">{plot_html}</div>'
        f'</section>'
    )
CHART_SECTIONS = "\n".join(section_parts)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Porsche 911 — Evolution Analysis</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:        #293241;
      --surface:   #1e2a38;
      --border:    #3d5a80;
      --text:      #e0fbfc;
      --muted:     #98c1d9;
      --accent:    #ee6c4d;
      --sidebar-w: 260px;
    }}

    body {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
    }}

    /* ── Sidebar ── */
    .sidebar {{
      position: fixed;
      top: 0; left: 0;
      width: var(--sidebar-w);
      height: 100vh;
      background: #1a2432;
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      z-index: 200;
      overflow-y: auto;
    }}

    .sidebar-top-bar {{
      height: 3px;
      background: var(--accent);
      flex-shrink: 0;
    }}

    .sidebar-brand {{
      padding: 22px 22px 18px;
      border-bottom: 1px solid var(--border);
    }}
    .brand-make {{
      font-size: 0.56rem;
      font-weight: 700;
      letter-spacing: 0.3em;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .brand-name {{
      font-size: 2rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.03em;
      line-height: 1;
      margin: 3px 0 10px;
    }}
    .brand-divider {{
      height: 1px;
      background: var(--border);
      margin-bottom: 9px;
    }}
    .brand-tag {{
      font-size: 0.57rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      color: var(--accent);
      text-transform: uppercase;
    }}

    .sidebar-nav {{ padding: 18px 0 8px; flex: 0 0 auto; }}
    .nav-section {{
      display: block;
      font-size: 0.56rem;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 0 22px 12px;
    }}
    .nav-list {{ list-style: none; }}
    .nav-link {{
      display: flex;
      align-items: center;
      gap: 11px;
      padding: 9px 22px;
      color: #5a7a90;
      text-decoration: none;
      font-size: 0.8rem;
      transition: color 0.18s;
    }}
    .nav-link::before {{
      content: '';
      width: 6px; height: 6px;
      border-radius: 50%;
      border: 1.5px solid #3d5a80;
      flex-shrink: 0;
      transition: background 0.18s, border-color 0.18s, transform 0.18s;
    }}
    .nav-link:hover {{ color: #98c1d9; }}
    .nav-link:hover::before {{
      border-color: var(--accent);
      transform: scale(1.25);
    }}
    .nav-link.active {{
      color: #fff;
      font-weight: 500;
    }}
    .nav-link.active::before {{
      background: var(--accent);
      border-color: var(--accent);
      transform: scale(1.15);
    }}

    .sidebar-stats {{
      margin: 6px 14px 14px;
      padding: 13px 14px;
      background: #1e2a38;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .stats-heading {{
      font-size: 0.55rem;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 7px;
    }}
    .stat-row:last-child {{ margin-bottom: 0; }}
    .stat-val {{
      font-size: 0.95rem;
      font-weight: 700;
      color: #fff;
      font-variant-numeric: tabular-nums;
    }}
    .stat-lbl {{
      font-size: 0.67rem;
      color: var(--muted);
    }}

    .sidebar-legend {{
      margin: 0 14px 14px;
      padding: 12px 14px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .legend-heading {{
      font-size: 0.55rem;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 9px;
    }}
    .legend-items {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 5px 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 0.7rem;
      color: var(--text);
    }}
    .legend-dot {{
      width: 8px; height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    .sidebar-footer {{
      margin-top: auto;
      padding: 11px 22px;
      border-top: 1px solid var(--border);
      font-size: 0.63rem;
      color: var(--muted);
      line-height: 1.7;
    }}

    /* ── Page wrap ── */
    .page-wrap {{ margin-left: var(--sidebar-w); }}

    /* ── Header ── */
    header {{
      background: linear-gradient(135deg, #1a2432 0%, #293241 65%, #1e2a38 100%);
      border-bottom: 1px solid rgba(61,90,128,0.4);
      padding: 52px 60px 44px;
      position: relative;
      overflow: hidden;
    }}
    header::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(ellipse at 78% 50%, rgba(238,108,77,0.15) 0%, transparent 60%);
      pointer-events: none;
    }}
    header::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--accent), transparent 60%);
    }}
    .header-eyebrow {{
      font-size: 0.65rem;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
      animation: slideIn 0.7s ease both;
    }}
    header h1 {{
      font-size: 2.8rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.01em;
      line-height: 1.1;
      animation: slideIn 0.7s 0.1s ease both;
    }}
    header h1 span {{ color: var(--accent); }}
    header p {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.95rem;
      animation: slideIn 0.7s 0.2s ease both;
    }}

    /* ── Main ── */
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 52px 44px 80px;
    }}

    /* ── Chart sections ── */
    .chart-section {{
      margin-bottom: 60px;
      opacity: 0;
      transform: translateY(24px);
      animation: fadeUp 0.55s ease forwards;
    }}
    .chart-header {{
      margin-bottom: 14px;
      padding-left: 14px;
      border-left: 3px solid var(--accent);
    }}
    .chart-header h2 {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 5px;
    }}
    .chart-desc {{
      font-size: 0.83rem;
      color: var(--muted);
      line-height: 1.65;
      max-width: 680px;
    }}
    .chart-body {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px;
      box-shadow: 0 4px 28px rgba(0,0,0,0.55);
      transition: border-color 0.25s, box-shadow 0.25s;
    }}
    .chart-body:hover {{
      border-color: rgba(238,108,77,0.4);
      box-shadow: 0 6px 32px rgba(238,108,77,0.12);
    }}

    @keyframes fadeUp {{
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes slideIn {{
      from {{ opacity: 0; transform: translateX(-16px); }}
      to   {{ opacity: 1; transform: translateX(0); }}
    }}
  </style>
</head>
<body>

  <aside class="sidebar">
    <div class="sidebar-top-bar"></div>
    <div class="sidebar-brand">
      <div class="brand-make">Porsche</div>
      <div class="brand-name">911</div>
      <div class="brand-divider"></div>
      <div class="brand-tag">Evolution Analysis</div>
    </div>
    <nav class="sidebar-nav">
      <span class="nav-section">Navigation</span>
      <ul class="nav-list">
        {nav_links}
      </ul>
    </nav>
    <div class="sidebar-stats">
      <div class="stats-heading">Dataset</div>
      <div class="stat-row">
        <span class="stat-val">{len(df)}</span>
        <span class="stat-lbl">variants</span>
      </div>
      <div class="stat-row">
        <span class="stat-val">{year_min}–{year_max}</span>
        <span class="stat-lbl">years</span>
      </div>
      <div class="stat-row">
        <span class="stat-val">{n_lap_matches}</span>
        <span class="stat-lbl">lap times matched</span>
      </div>
    </div>
    <div class="sidebar-legend">
      <div class="legend-heading">Generations</div>
      <div class="legend-items">
        {legend_items_html}
      </div>
    </div>
    <div class="sidebar-footer">
      Kaggle · fastestlaps.com
    </div>
  </aside>

  <div class="page-wrap">
    <header>
      <div class="header-eyebrow">Performance Analysis</div>
      <h1>The Porsche <span>911</span></h1>
      <p>60+ years of flat-six engineering — data driven</p>
    </header>
    <main>
      {CHART_SECTIONS}
    </main>
  </div>

  <script>
    const sections = document.querySelectorAll('.chart-section');
    const links    = document.querySelectorAll('.nav-link');

    // Fade-in animations: trigger when section enters viewport
    const fadeIO = new IntersectionObserver((entries) => {{
      entries.forEach(e => {{
        if (e.isIntersecting) e.target.style.animationPlayState = 'running';
      }});
    }}, {{ threshold: 0.05 }});

    sections.forEach(s => {{
      s.style.animationPlayState = 'paused';
      fadeIO.observe(s);
    }});

    // Active nav link: find the section whose top is closest to (but above) 40% viewport height
    function updateActive() {{
      let current = '';
      sections.forEach(s => {{
        if (s.getBoundingClientRect().top <= window.innerHeight * 0.4) current = s.id;
      }});
      links.forEach(l => l.classList.remove('active'));
      if (current) {{
        const a = document.querySelector('.nav-link[href="#' + current + '"]');
        if (a) a.classList.add('active');
      }}
    }}

    window.addEventListener('scroll', updateActive, {{ passive: true }});
    updateActive();
  </script>
</body>
</html>"""

# save the finished HTML file next to this script, then open it in the browser
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
print(f"Dashboard saved → {out_path}")

