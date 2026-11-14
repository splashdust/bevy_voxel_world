#!/bin/bash
# Throwaway: builds chunk_lifecycle in the probe worktree of main with the
# colour cost simulation patched in (colour_cost_sim.patch: a solver that
# spreads a run per colour group or per component under COLOUR_SIM, landing
# that reads two more components per border voxel, the edits_lamps scenario,
# and LAMP_REACH / LAMP_LEVELS on the lamp world), copies the binary to
# target/colour_bins/chunk_lifecycle_sim, and reverts the patch. Exits non-zero
# if the build fails.
cd "$(dirname "$0")/../coloured-light/probe-tree" || exit 1
git apply ../../probes/colour_cost_sim.patch || exit 1
cargo bench --features light_propagation --bench chunk_lifecycle --no-run 2>&1 \
  | grep -E "^error|^warning|^ +--> |Finished|Executable|could not compile|^  *= |^[0-9 ]+\|" | head -150
result=${PIPESTATUS[0]}
binary=$(ls -t target/release/deps/chunk_lifecycle-* 2>/dev/null | grep -v '\.d$' | head -1)
git apply -R ../../probes/colour_cost_sim.patch
[ $result -eq 0 ] || exit $result
mkdir -p target/colour_bins
cp "$binary" target/colour_bins/chunk_lifecycle_sim && echo "copied $binary"
