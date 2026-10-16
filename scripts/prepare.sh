#!/usr/bin/env bash
# 把 data/raw/*.jpg 批量缩放为 64x64 PPM 到 data/ppm/
set -eu
SIZE="${1:-64}"
mkdir -p data/ppm
n=0
for f in data/raw/*.jpg; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .jpg)"
  ./c/img2ppm "$f" "data/ppm/$base.ppm" "$SIZE" && n=$((n+1))
done
echo "converted $n images to data/ppm (${SIZE}x${SIZE})"
