HINN CONFERENCE SOURCE - ESRI MAP FIXED EDITION
===============================================

START
-----
1. Extract the ZIP.
2. Open OPEN_ME.html.
3. Click OPEN 20 AIS CASES - GEOGRAPHIC MAP.

BASEMAP
-------
Default:
  Esri World Street Map

Automatic fallback:
  Esri World Topographic Map

Optional manual fallback:
  OpenStreetMap Standard

CARTO is intentionally NOT used in this package because current unauthenticated
CARTO raster basemaps display an "API KEY REQUIRED" watermark.

No Leaflet/CDN JavaScript is needed.
Internet is needed only for the geographic basemap.
All HINN/RF/RNN/LSTM trajectories and local comparison plots are stored locally.
Only visible cases request map tiles (lazy rendering).

SOURCE
------
source/generate_20case_map_html.py
data/Selected20Case_GeoreferencedTrajectories.json
source/verify_map_package.py
