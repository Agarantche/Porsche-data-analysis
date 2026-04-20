# Porsche-data-analysisPorsche 911 Evolution Analysis

A data analysis and interactive dashboard exploring 60+ years of Porsche 911 engineering, built with Python and Plotly.

What it does:
Loads a dataset of ~300 Porsche 911 variants from Kaggle, scrapes live Nürburgring lap times from fastestlaps.com, and generates a single-page interactive dashboard with 6–8 charts covering:

A custom Driver Score ranking every variant by (HP/Weight) ÷ 0–60 time
Weight vs performance sweet spot analysis
Engine efficiency (HP/litre) by generation
Air-cooled vs water-cooled era comparison
1970s–80s emissions penalty — how much power US-spec cars lost to catalytic converters
HP/kg trend over time with a smoothed trendline
Nürburgring Score using real lap times (HP/kg) ÷ lap seconds for matched cars
Rear track width vs lap time correlation
Stack: Python · pandas · Plotly · BeautifulSoup · thefuzz

Run porsche_analysis.py — it scrapes the lap data, builds all charts, and opens porsche_dashboard.html in your browser automatically.

Dataset: Every Porsche 911 — Kaggle
