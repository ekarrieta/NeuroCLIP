#!/usr/bin/env bash
#SBATCH --job-name=neuroclip-eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

features=(wav2vec2 envelope mfcc mel_spectrogram)
models=(cnn cnn conformer conformer)

index="${SLURM_ARRAY_TASK_ID}"
feature="${features[${index}]}"
model="${models[${index}]}"
train_job="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID}"
run_name="grid-${feature}-${model}-${train_job}-${index}"

python eval_model.py \
  --checkpoint-dir "checkpoints/gwilliams2022/${run_name}" \
  --split test \
  --batch-size 1000 \
  --num-workers "${SLURM_CPUS_PER_TASK}"
