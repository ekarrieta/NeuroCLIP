#!/usr/bin/env bash
#SBATCH --job-name=neuroclip-train
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

features=(wav2vec2 envelope mfcc mel_spectrogram)
models=(cnn cnn conformer conformer)

index="${SLURM_ARRAY_TASK_ID}"
feature="${features[${index}]}"
model="${models[${index}]}"
run_name="grid-${feature}-${model}-${SLURM_ARRAY_JOB_ID}-${index}"

python neuroclip.py \
  --dataset gwilliams2022 \
  --name "${run_name}" \
  --feature-type "${feature}" \
  --model "${model}" \
  --batch-size 256 \
  --num-workers "${SLURM_CPUS_PER_TASK}"
