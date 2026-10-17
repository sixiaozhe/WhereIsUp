#!/usr/bin/env bash
# 校验 numpy 与 C 的前向 logits 一致（最大绝对差 < 1e-4）。
# 同时覆盖默认的四方向 TTA 与 --no-tta 单次前向。
set -eu
TOL="1e-4"
n="${1:-8}"
pass=0
i=0
for f in data/ppm/*.ppm; do
  [ "$i" -lt "$n" ] || break
  for mode in tta single; do
    if [ "$mode" = "single" ]; then extra="--no-tta"; else extra=""; fi
    # shellcheck disable=SC2086
    py=$(PYTHONPATH=src python3 src/infer_py.py "$f" --logits $extra)
    # shellcheck disable=SC2086
    cc=$(./c/whereisup --ppm "$f" --logits $extra)
    PY_OUT="$py" CC_OUT="$cc" TOL="$TOL" MODE="$mode" python3 - <<'PY'
import os, sys
py = [float(v) for v in os.environ["PY_OUT"].split()]
cc = [float(v) for v in os.environ["CC_OUT"].split()]
tol = float(os.environ["TOL"])
d = max(abs(a-b) for a, b in zip(py, cc))
if d >= tol:
    print(f"MISMATCH [{os.environ['MODE']}] max|Δ|={d:.3e} py={py} cc={cc}")
    sys.exit(1)
print(f"ok [{os.environ['MODE']}] max|Δ|={d:.3e}")
PY
    pass=$((pass+1))
  done
  i=$((i+1))
done
echo "parity passed for $pass checks ($i images x 2 modes, tol $TOL)"
