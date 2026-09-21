#!/usr/bin/env python3
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/"data"/"Selected20Case_GeoreferencedTrajectories.json").read_text(encoding="utf-8"))
html = (ROOT/"cases20"/"MAP_20_CASES.html").read_text(encoding="utf-8")

errors = []
if len(data) != 20:
    errors.append(f"Expected 20 cases, got {len(data)}")

for c in data:
    p = ROOT/"cases20"/c["local_img"]
    if not p.exists():
        errors.append(f"Missing local comparison: {p}")
    for k in ["history","truth","rf","rnn","lstm","hinn"]:
        if not c.get(k):
            errors.append(f"Case {c.get('case')} missing {k}")

if "World_Street_Map/MapServer/tile" not in html:
    errors.append("Esri World Street Map provider missing")
if "World_Topo_Map/MapServer/tile" not in html:
    errors.append("Esri World Topographic fallback missing")
if "basemaps.cartocdn.com" in html or "CARTO Voyager" in html:
    errors.append("CARTO references still present")
if "IntersectionObserver" not in html:
    errors.append("Lazy rendering missing")

if errors:
    print("VERIFY: FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("VERIFY: PASS")
print("20 georeferenced cases present.")
print("Esri World Street Map default present.")
print("Esri World Topographic fallback present.")
print("No CARTO API-key-dependent basemap remains.")
print("Lazy rendering present.")
print("All local comparison images present.")
