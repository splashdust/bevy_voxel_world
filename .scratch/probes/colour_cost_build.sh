#!/bin/bash
# Throwaway: builds the coloured-light cost probe in the probe worktree of main
# (.scratch/coloured-light/probe-tree) with the bench profile, its light_bench
# accessors patched in for the build only, leaving no file in its examples/.
# Exits non-zero if the build fails, so that a script can't go on with an old binary.
cd "$(dirname "$0")/../coloured-light/probe-tree" || exit 1
git apply ../../probes/colour_cost_bench.patch || exit 1
cp ../../probes/colour_cost.rs examples/colour_cost_01.rs
cargo build --profile bench --features light_propagation --example colour_cost_01 2>&1 | grep -E "^error|^warning|^ +--> |Finished|could not compile|^  *= |^[0-9 ]+\|" | head -150
result=${PIPESTATUS[0]}
rm -f examples/colour_cost_01.rs
git apply -R ../../probes/colour_cost_bench.patch
exit $result
