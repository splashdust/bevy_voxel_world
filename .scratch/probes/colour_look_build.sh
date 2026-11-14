#!/bin/zsh
# Throwaway: builds the coloured-light look probe in the probe worktree of main
# (.scratch/coloured-light/probe-tree) without leaving a file in its examples/.
# Exits non-zero if the build fails, so that a script can't go on with an old binary.
cd "$(dirname "$0")/../coloured-light/probe-tree" || exit 1
cp ../../probes/colour_look.rs examples/colour_look_01.rs
cargo build --release --features light_propagation --example colour_look_01 2>&1 | grep -E "^error|^warning: unused|^ +--> |Finished|could not compile|^  *= |^[0-9 ]+\|" | head -150
result=${pipestatus[1]}
rm -f examples/colour_look_01.rs
exit $result
