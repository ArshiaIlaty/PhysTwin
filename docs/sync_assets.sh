#!/usr/bin/env bash
# Copy latest experiment figures into docs/assets/ for GitHub Pages.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$ROOT/docs"

echo "Syncing Arshia figures..."
mkdir -p "$DOCS/assets/arshia" "$DOCS/assets/phystwin" "$DOCS/assets/malak"

if [[ -d "$ROOT/arshia_work/forward_force_results/figures" ]]; then
  cp -f "$ROOT/arshia_work/forward_force_results/figures/"*.png "$DOCS/assets/arshia/" 2>/dev/null || true
fi

if [[ -d "$ROOT/arshia_work/forward_force_results/videos" ]]; then
  cp -f "$ROOT/arshia_work/forward_force_results/videos/"*.mp4 "$DOCS/assets/arshia/" 2>/dev/null || true
fi

echo "Syncing PhysTwin teaser/GIFs..."
for f in teaser.png force_cloth.gif force_rope.gif force_sloth.gif; do
  if [[ -f "$ROOT/assets/$f" ]]; then
    cp -f "$ROOT/assets/$f" "$DOCS/assets/phystwin/"
  fi
done

echo "Done. Assets in docs/assets/:"
find "$DOCS/assets" -type f | sort
