#!/usr/bin/env bash
# V2 全自动流水线：80k 训练（单余弦/batch64）→ 4ckpt×4套件全量评测 → 汇总
set -u
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
cd /root/autodl-tmp/src/lerobot

META=/root/autodl-tmp/outputs/_runmeta/v2
OUT=/root/autodl-tmp/outputs/libero_smolvla_v2
VLM=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct
mkdir -p "$META"

train() {
  local BS=$1
  lerobot-train \
    --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
    --policy.vlm_model_name=$VLM \
    --policy.input_features=null --policy.output_features=null \
    --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
    --policy.scheduler_decay_steps=80000 \
    --dataset.repo_id=lerobot/libero \
    --dataset.root=/root/autodl-tmp/SmolVLA/libero \
    --dataset.video_backend=pyav \
    --output_dir=$OUT --job_name=libero_smolvla_v2 \
    --batch_size=$BS --steps=80000 --save_freq=10000 --log_freq=50 \
    --num_workers=16 --env_eval_freq=0 --seed=1000
}

echo "===== TRAIN START batch=64 $(date) ====="
train 64
RC=$?
if [ $RC -ne 0 ] && grep -qiE "out of memory|CUDA error" "$META/pipeline.log"; then
  echo "===== OOM 回退 batch=48 重试 ====="
  rm -rf "$OUT"
  train 48
  RC=$?
fi
echo "TRAIN_EXIT=$RC" > "$META/train.exit"
[ $RC -ne 0 ] && { echo "训练失败，流水线终止"; exit $RC; }

echo "===== EVAL START $(date) ====="
for CK in 020000 040000 060000 080000; do
  for S in libero_spatial libero_object libero_goal libero_10; do
    EOUT=/root/autodl-tmp/outputs/eval_v2_${CK}_${S}
    [ -f "$EOUT/eval_info.json" ] && { echo "SKIP $CK/$S"; continue; }
    echo "===== EVAL $CK $S $(date) ====="
    lerobot-eval \
      --policy.path=$OUT/checkpoints/$CK/pretrained_model \
      --policy.vlm_model_name=$VLM \
      --env.type=libero --env.task=$S \
      --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
      --output_dir=$EOUT
    echo "EVAL_${CK}_${S}_EXIT=$?" >> "$META/eval.exit"
  done
done

echo "===== SUMMARIZE $(date) ====="
python3 - <<'PY'
import json
res = {}
for ck in ["020000","040000","060000","080000"]:
    row = {}
    for s in ["libero_spatial","libero_object","libero_goal","libero_10"]:
        try:
            d = json.load(open(f"/root/autodl-tmp/outputs/eval_v2_{ck}_{s}/eval_info.json"))
            row[s] = d["overall"]["pc_success"]
        except Exception:
            row[s] = None
    vals = [v for v in row.values() if v is not None]
    row["AVG"] = round(sum(vals)/len(vals), 1) if vals else None
    res[ck] = row
meta = "/root/autodl-tmp/outputs/_runmeta/v2"
json.dump(res, open(f"{meta}/summary.json","w"), indent=1)
best = max(res, key=lambda k: res[k]["AVG"] or -1)
lines = ["| ckpt | spatial | object | goal | libero_10 | AVG |",
         "|---|---|---|---|---|---|"]
for ck, r in res.items():
    fmt = lambda v: f"{v}" if v is not None else "—"
    star = " ⭐" if ck == best else ""
    lines.append(f"| {ck}{star} | {fmt(r['libero_spatial'])} | {fmt(r['libero_object'])} | {fmt(r['libero_goal'])} | {fmt(r['libero_10'])} | {fmt(r['AVG'])} |")
open(f"{meta}/table.md","w").write("\n".join(lines))
print(json.dumps(res, indent=1))
print("BEST:", best)
PY
echo DONE > "$META/all.done"
echo "===== PIPELINE COMPLETE $(date) ====="
