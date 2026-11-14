#!/usr/bin/env bash
# V2 评测续跑（只评测，不重训）：060000 -> 080000 四套，标准 400-episode 口径
# 与 v2_pipeline.sh 的评测段参数完全一致，保证与 20k/40k 结果可比
set -u
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
cd /root/autodl-tmp/src/lerobot

META=/root/autodl-tmp/outputs/_runmeta/v2
OUT=/root/autodl-tmp/outputs/libero_smolvla_v2
VLM=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct
mkdir -p "$META"

echo "===== EVAL RESUME START $(date) ====="
for CK in 060000 080000; do
  for S in libero_spatial libero_object libero_goal libero_10; do
    EOUT=/root/autodl-tmp/outputs/eval_v2_${CK}_${S}
    if [ -f "$EOUT/eval_info.json" ]; then
      echo "SKIP $CK/$S (eval_info.json exists)"
      continue
    fi
    echo "===== EVAL $CK $S $(date) ====="
    lerobot-eval \
      --policy.path=$OUT/checkpoints/$CK/pretrained_model \
      --policy.vlm_model_name=$VLM \
      --env.type=libero --env.task=$S \
      --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
      --output_dir=$EOUT
    RC=$?
    echo "EVAL_${CK}_${S}_EXIT=$RC" | tee -a "$META/eval.exit"
    if [ $RC -ne 0 ]; then echo "WARN eval failed: $CK/$S rc=$RC"; fi
  done
done

echo "===== SUMMARIZE $(date) ====="
python3 - <<'PY'
import json
meta = "/root/autodl-tmp/outputs/_runmeta/v2"
res = {}
for ck in ["020000", "040000", "060000", "080000"]:
    row = {}
    for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        try:
            d = json.load(open(f"/root/autodl-tmp/outputs/eval_v2_{ck}_{s}/eval_info.json"))
            row[s] = d["overall"]["pc_success"]
        except Exception:
            row[s] = None
    vals = [v for v in row.values() if v is not None]
    row["AVG"] = round(sum(vals) / len(vals), 1) if vals else None
    res[ck] = row
json.dump(res, open(f"{meta}/summary.json", "w"), indent=1)
best = max(res, key=lambda k: res[k]["AVG"] or -1)
lines = ["| ckpt | spatial | object | goal | libero_10 | AVG |", "|---|---|---|---|---|---|"]
for ck, r in res.items():
    fmt = lambda v: f"{v}" if v is not None else "-"
    star = " *" if ck == best else ""
    lines.append(f"| {ck}{star} | {fmt(r['libero_spatial'])} | {fmt(r['libero_object'])} | "
                 f"{fmt(r['libero_goal'])} | {fmt(r['libero_10'])} | {fmt(r['AVG'])} |")
open(f"{meta}/table.md", "w").write("\n".join(lines))
print("\n".join(lines))
print("BEST:", best)
PY
echo "===== EVAL RESUME COMPLETE $(date) ====="
echo DONE > "$META/all.eval.done"
