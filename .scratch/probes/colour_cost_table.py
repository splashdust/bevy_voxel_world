"""Throwaway: tabulates the colour cost lifecycle runs in
.scratch/coloured-light/baseline/lifecycle/ (one .txt per label) for the
coloured-light map's ticket "What each colour candidate costs"."""
import re
import sys
from pathlib import Path

folder = Path(sys.argv[1])
keys = [
    "edits_lamps.p50_frames_to_land",
    "edits_lamps.max_frames_to_land",
    "edits_lamps.gate_failures",
    "edits_lamps.p50_main_thread_us",
    "edits_lamps.max_main_thread_us",
    "edits_lamps.p50_relights",
    "edits_lamps.max_relights",
    "edits_lamps.relight_us_per_relight",
    "stream_in_lamps.time_ms",
    "stream_in_lamps_unlit.time_ms",
    "stream_in_lamps.rest_ms",
    "stream_in_lamps.pool_solve_us_per_chunk",
    "stream_in_lamps.main_thread_max_frame_ms",
    "caves.gate_failures",
    "caves.capped",
    "caves.solves_per_arrival",
    "caves.visible_per_shown",
    "caves.pool_solve_us_per_chunk",
    "caves.relight_us_per_relight",
    "caves.time_ms",
    "caves.settled_ms",
    "caves.arrivals",
    "caves.peak_slots",
    "caves.main_thread_max_frame_ms",
]
rows = {}
frames = {}
for path in sorted(folder.glob("cl*.txt"), key=lambda path: path.stat().st_mtime):
    text = path.read_text()
    metrics = {}
    for line in text.splitlines():
        match = re.match(r"^(\S+)\s+(-?[\d.]+|NaN|inf)$", line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)
    rows[path.stem] = metrics
    lists = re.findall(r"edits_lamps: .*?frames \[([^\]]*)\]", text)
    frames[path.stem] = [l.replace(".0", "") for l in lists]
labels = list(rows)
width = max(len(k) for k in keys)
print(f"{'':{width}} " + " ".join(f"{l[4:]:>12}" for l in labels))
for key in keys:
    print(f"{key:{width}} " + " ".join(f"{float(rows[l].get(key, 'nan')):>12.1f}" for l in labels))
print()
for label in labels:
    distinct = sorted(set(frames[label]))
    print(label, "edit frames per run:", " | ".join(distinct) if distinct else "-")
