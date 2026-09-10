# NeuroCLIP

NeuroCLIP learns a time-aligned representation of speech from
magnetoencephalography (MEG). A convolutional or Conformer MEG encoder is
trained with a CLIP-style contrastive objective against acoustic, speech-model,
or text representations. The repository contains the training and retrieval
evaluation paths, reusable model layers, dataset loaders, preprocessing tools,
configuration examples, and portable Slurm array examples.

## Installation

Python 3.10 or newer and an NVIDIA GPU are recommended. Create an isolated
environment and install the pinned dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Wav2Vec2 weights are downloaded from Hugging Face on first use. Two feature
families need extra local configuration:

    export OPENAI_API_KEY=...                  # word/sentence embeddings
    export FASTTEXT_MODEL_PATH=/path/model.bin # static embeddings

Credentials, datasets, caches, checkpoints, logs, and generated results are
ignored by Git.

## Public datasets

NeuroCLIP currently supports two natural-speech MEG datasets:

- [MEG-MASC (Gwilliams et al.)](https://osf.io/ag3kj/) is released through OSF
  under CC0. Download its BIDS release into data/gwilliams2022. The
  [dataset paper](https://doi.org/10.1038/s41597-023-02752-5) describes the
  recordings and annotations.
- [MOUS (Schoffelen et al.)](https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236)
  is distributed through the Donders Repository. Access requires an account
  and acceptance of its data-use agreement. Place a BIDS-formatted copy of the
  auditory MEG data and stimuli in data/schoffelen2019. See the
  [dataset paper](https://doi.org/10.1038/s41597-019-0020-y).

The default locations and acquisition settings live in readers/gwilliams2022.yaml
and readers/schoffelen2019.yaml. Use --bids-root and --stim-root to keep large
datasets elsewhere.

## Preprocessing

For MEG-MASC, generate word events, content-safe splits, and three-second
segments with:

    python -m readers.gwilliams2022.prepare --lengths 3.0

For MOUS, event extraction is followed by Dutch forced alignment and segment
creation. The exact phases, expected schema, split conventions, and all
preprocessing options are documented in [readers/README.md](readers/README.md).

Both pipelines write the table expected by training to:

    cache/<dataset>/annotations/segments_<length>s.tsv

On first access, readers/dataset.py resamples continuous MEG to 120 Hz and audio
to 16 kHz, stores per-recording memory maps under cache, applies a 0.5-second
within-window baseline correction, and returns fixed-size tensors plus segment
metadata. Start the first run with a single data-loading worker if several jobs
would otherwise build the same cache concurrently.

## Training

A minimal run is:

    python neuroclip.py \
      --name gwilliams-wav2vec2-cnn \
      --dataset gwilliams2022 \
      --length 3.0 \
      --model cnn \
      --feature-type wav2vec2

A YAML configuration can provide defaults; explicit flags still win:

    python neuroclip.py --config configs/train_example.yaml --epochs 30

The feature choices are:

| Feature | Representation |
| --- | --- |
| wav2vec2 | Mean of Wav2Vec2 hidden layers 14 through 18 |
| wav | Waveform resampled onto the MEG time grid |
| mfcc | 13 MFCCs projected to the model feature width |
| mel_spectrogram | Log-mel spectrum projected to the model feature width |
| vad | Smoothed energy-based voice activity |
| envelope | Smoothed Hilbert amplitude envelope |
| pitch | CREPE fundamental-frequency estimate |
| rms | Short-time root-mean-square energy |
| zcr | Smoothed zero-crossing rate |
| word_embeddings | OpenAI embeddings distributed across words |
| sentence_embeddings | One OpenAI embedding repeated over the window |
| anchor_word | OpenAI embedding for the segment's anchor word |
| static_embeddings | Mean fastText word vector |
| flipped_wav2vec2 | Time-reversed control for MEG and audio |
| noise | Gaussian-noise control |

The main training parameters are:

| Option | Meaning | Default |
| --- | --- | --- |
| --dataset | gwilliams2022 or schoffelen2019 | gwilliams2022 |
| --dataset-config | Dataset YAML | readers/<dataset>.yaml |
| --segments | Precomputed segment TSV | Derived from dataset and length |
| --bids-root, --stim-root | Override data locations | Dataset YAML |
| --cache-dir, --output-dir | Generated cache and run roots | cache, checkpoints |
| --length | Window length in seconds | 3.0 |
| --model | cnn or conformer | cnn |
| --feature-type | Target representation listed above | wav2vec2 |
| --epochs, --early-stop | Maximum epochs and patience | 20, 3 |
| --learning-rate, --weight-decay | Adam optimizer settings | 3e-4, 0 |
| --batch-size, --eval-batch-size | Train and retrieval batch sizes | 256, 1000 |
| --num-workers, --pin-memory | DataLoader settings | 3, automatic |
| --seed, --device | Reproducibility and PyTorch device | 2036, auto |
| --dropout | Conformer dropout | 0.2 |
| --num-layers, --hidden, --num-heads | Conformer dimensions | 6, 320, 8 |
| --initial-linear, --meg-hidden | Override model YAML widths | Model YAML |
| --clamp | Normalized MEG clipping limit | 20 |
| --scaler-meg-samples | Samples fitted per recording | 200 |
| --scaler-feature-samples | Feature samples used for fitting | 8000 |
| --scaler-batch-size | Scaler fitting batch size | 16 |
| --eval-negatives | Sampled negatives per query | Full evaluation batch |
| --no-final-eval | Skip test retrieval after training | Evaluation enabled |
| --wav2vec-model, --openai-model | Override upstream model identifiers | See help |

Run python neuroclip.py --help for aliases and the complete generated help.

Each run is self-contained under checkpoints/<dataset>/<name>:

- best_model.pth: best validation checkpoint.
- run_config.yaml: CLI arguments, resolved paths, and exact model settings.
- history.csv: one row per epoch with train and validation losses.
- metrics.json: dataset sizes, timing, best epoch/loss, and test metrics.
- test_queries.tsv: one row per query with rank, retrieval correctness,
  identifiers, text metadata, scores, and score margin.
- train.log: readable console history.

These files support aggregate comparisons and post-hoc error analysis with
ordinary tabular and JSON tooling.

## Evaluation

Standalone evaluation automatically reads architecture and feature defaults
from the checkpoint:

    python eval_model.py \
      --checkpoint-dir checkpoints/gwilliams2022/gwilliams-wav2vec2-cnn

Or start from the example configuration:

    python eval_model.py --config configs/eval_example.yaml

Evaluation parameters are:

| Option | Meaning | Default |
| --- | --- | --- |
| --checkpoint-dir | Directory containing best_model.pth | Required |
| --dataset, --model, --feature-type, --length | Override checkpoint metadata | Checkpoint |
| --dataset-config, --segments | Override dataset YAML or segment TSV | Derived |
| --bids-root, --stim-root, --cache-dir | Override local data paths | Dataset defaults |
| --split | train, valid, or test | test |
| --batch-size, --num-workers | Candidate batch and loader workers | 1000, 3 |
| --negatives | Random negatives per query | Every other batch item |
| --seed | Negative-sampling seed | 2036 |
| --shuffle | Shuffle before making candidate batches | Disabled |
| --output-dir | Result destination | checkpoint/evaluation/<split> |
| --overwrite | Reuse that destination | New numeric suffix |
| --device | auto, cpu, cuda, or a CUDA index | auto |
| architecture options | Manual overrides for legacy checkpoints | Checkpoint |
| --clamp | Normalized MEG clipping limit | Checkpoint or 20 |
| feature model options | Override Wav2Vec2/OpenAI identifiers | Checkpoint |

Evaluation writes metrics.json and queries.tsv. Top-1, top-10, mean reciprocal
rank, and contrastive loss are reported. Duplicate stimulus IDs are treated as
valid positives. Retrieval candidates come from each batch, so batch size and
ordering are part of the evaluation protocol and are recorded in the output.

## Slurm arrays

Generic array jobs are in examples/slurm. Adjust resources for the cluster,
then create the ignored log directory and submit:

    mkdir -p logs
    sbatch examples/slurm/train_grid.sbatch

The training job ID is included in run names. Evaluate the matching grid with:

    sbatch --export=ALL,TRAIN_ARRAY_JOB_ID=<job-id> examples/slurm/eval_grid.sbatch

The launchers contain no cluster-specific paths, accounts, partitions, or email
addresses.

## Repository layout

- neuroclip.py: training CLI and local experiment records.
- eval_model.py: retrieval metrics and query-level exports.
- features.py: acoustic, Wav2Vec2, OpenAI, and fastText features.
- forward.py and norm.py: batch processing and fitted normalization.
- models/: CNN, Conformer, spatial channel layers, and contrastive losses.
- readers/: portable BIDS loading, dataset YAML, and preprocessing.
- configs/: runnable YAML examples.
- examples/slurm/: portable array-launch templates.

## Attribution and license

The convolutional MEG backbone, spatial channel components, and contrastive
loss build on [BrainMagick](https://github.com/facebookresearch/brainmagick) and
the work of Défossez et al. Attribution and licensing details are recorded in
[LICENSE](LICENSE).

NeuroCLIP is distributed under CC BY-NC 4.0; see [LICENSE](LICENSE).
