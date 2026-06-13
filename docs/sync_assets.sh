#!/usr/bin/env bash
# Copy latest experiment figures into docs/assets/ for GitHub Pages.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$ROOT/docs"

echo "Syncing scenario figures..."
mkdir -p "$DOCS/assets/scenario_sweeps" "$DOCS/assets/phystwin" "$DOCS/assets/force_control"

if [[ -f "$ROOT/scenario_sweeps/plot_scenario_figures.py" ]]; then
  python "$ROOT/scenario_sweeps/plot_scenario_figures.py" --out_dir "$DOCS/assets/scenario_sweeps" || true
fi

if [[ -d "$ROOT/scenario_sweeps/forward_force_results/figures" ]]; then
  cp -f "$ROOT/scenario_sweeps/forward_force_results/figures/"*.png "$DOCS/assets/scenario_sweeps/" 2>/dev/null || true
fi

if [[ -d "$ROOT/scenario_sweeps/forward_force_results/videos" ]]; then
  echo "Converting scenario videos to H.264 for browser playback..."
  python "$DOCS/convert_videos_for_web.py" --overwrite || true
fi

echo "Syncing PhysTwin teaser/GIFs..."
for f in teaser.png force_cloth.gif force_rope.gif force_sloth.gif; do
  if [[ -f "$ROOT/assets/$f" ]]; then
    cp -f "$ROOT/assets/$f" "$DOCS/assets/phystwin/"
  fi
done

echo "Done. Assets in docs/assets/:"
find "$DOCS/assets" -type f | sort
