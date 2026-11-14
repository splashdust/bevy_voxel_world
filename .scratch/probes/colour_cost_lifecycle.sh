#!/bin/bash
# Throwaway: runs chunk_lifecycle's lit scenarios with the colour cost
# simulation (colour_cost_sim.patch) for the coloured-light map's ticket
# "What each colour candidate costs". Takes the patched bench binary, built by
# colour_cost_sim_build.sh, and a round number for labels. Results land in the
# probe worktree's target/bench/chunk_lifecycle/ as cl<round>-<label>.json, and
# each run's output in .scratch/coloured-light/baseline/lifecycle/.
bin="$1"
round="$2"
out="$(dirname "$0")/../coloured-light/baseline/lifecycle"
mkdir -p "$out"
run() {
  local label="$1"; shift
  local scenarios="$1"; shift
  echo "=== cl${round}-${label}: $scenarios ($*)" >&2
  env "$@" "$bin" run "cl${round}-${label}" "$scenarios" > "$out/cl${round}-${label}.txt" 2>&1
  echo "    exit $?" >&2
}
run white edits_lamps,caves,stream_in_lamps,stream_in_lamps_unlit,edits_lit COLOUR_SIM=off
run groups3 edits_lamps,caves COLOUR_SIM=groups3
run groups8 edits_lamps,caves COLOUR_SIM=groups8
run components edits_lamps,caves COLOUR_SIM=components
for sim in off groups8 components; do
  run "r32-$sim" edits_lamps,caves COLOUR_SIM=$sim LAMP_REACH=32
done
for sim in off groups8 components; do
  run "l256-$sim" edits_lamps,caves COLOUR_SIM=$sim LAMP_LEVELS=256
done
for sim in off groups8 components; do
  run "r32l256-$sim" edits_lamps COLOUR_SIM=$sim LAMP_REACH=32 LAMP_LEVELS=256
done
echo "done" >&2
