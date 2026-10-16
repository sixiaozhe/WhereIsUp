#!/usr/bin/env bash
# 从 picsum.photos 下载“正确朝向”的参考照片（Unsplash 授权）。
# 用法: scripts/download.sh [COUNT] [SIZE]
set -u
COUNT="${1:-2000}"
SIZE="${2:-400}"
OUT="data/raw"
mkdir -p "$OUT"
ok=0
for ((id=0; id<=1084 && ok<COUNT; id++)); do
  f="$OUT/$(printf '%04d' "$id").jpg"
  if [ -s "$f" ]; then ok=$((ok+1)); continue; fi
  url="https://picsum.photos/id/$id/$SIZE/$SIZE.jpg"
  if curl -fsSL --max-time 30 -o "$f" "$url" && [ -s "$f" ]; then
    magic=$(od -An -tx1 -N2 "$f" | tr -d ' \n')
    if [ "$magic" = "ffd8" ]; then
      ok=$((ok+1))
    else
      rm -f "$f"
    fi
  else
    rm -f "$f"
  fi
done
echo "done: $ok photos in $OUT (scanned id 0..$id)"
