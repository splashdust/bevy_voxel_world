#!/bin/bash
# Throwaway: builds the coloured-light carried-index probe in the probe worktree
# of main (.scratch/coloured-light/probe-tree) with the bench profile, the
# kernel and light_bench patched to carry a colour index (colour_label.patch)
# for the build only, leaving no file in its examples/.
# Exits non-zero if the build fails, so that a script can't go on with an old binary.
cd "$(dirname "$0")/../coloured-light/probe-tree" || exit 1
git apply ../../probes/colour_label.patch || exit 1
cp ../../probes/colour_label.rs examples/colour_label_01.rs
cargo build --profile bench --features light_propagation --example colour_label_01 2>&1 | grep -E "^error|^warning|^ +--> |Finished|could not compile|^  *= |^[0-9 ]+\|" | head -150
result=${PIPESTATUS[0]}
rm -f examples/colour_label_01.rs
git apply -R ../../probes/colour_label.patch
exit $result
