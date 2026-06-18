# ReaEmb: Reasoning-based Embedding Generator for Sequential Recommendation
This is the implementation of the paper "Harmonizing Semantic and Collaborative in LLMs: Reasoning-based Embedding Generator for Sequential Recommendation".

ReaEmb contains a two-stage LLM-based item embedding generator and a downstream sequential recommendation framework:
- **LRCL**: latent reasoning-enhanced contrastive learning for semantic item embedding generation.
- **CRRL**: collaborative reward reinforcement learning for injecting item co-occurrence signals.
- **Downstream SR**: training SASRec / GRU4Rec / BERT4Rec with the generated item embeddings.

## Configure the Environment
To facilitate reproducibility, below are the hardware and software settings used in our experiments.

**Hardware**
- GPU: `4 x NVIDIA GeForce RTX 3090 24GB`
- CPU: `AMD EPYC 7543 32-Core Processor`

**Software**
- Python: `3.9.5`
- PyTorch: `2.7.1+cu118`

Install Python dependencies via:

`pip install -r requirements.txt`

## Dataset Preprocessing & Embedding Generation

Follow the steps below to prepare data and produce ReaEmb item embeddings.

### 1) Prepare datasets

The processed datasets used by the training scripts are placed under `data/`:

```text
data/<yelp/CD/games>/handled/
├─ inter.txt
└─ item_info.jsonline
```

- `inter.txt`: processed user-item interaction sequences.
- `item_info.jsonline`: item textual descriptions for the LLM embedding generator.

Supported datasets:
- **Yelp**: from the Yelp Open Dataset.
- **CD**: from the Amazon review dataset.
- **Games**: from the Amazon review dataset.

For Amazon CD and Amazon Games, raw-data preprocessing scripts are also provided:

```text
data/<CD/games>/dataset/
├─ data_process.py
└─ item_prompt.py
```

Place the raw Amazon review and metadata files in the corresponding `dataset/` directory, then run:

```bash
cd data/<CD/games>/dataset
python data_process.py
python item_prompt.py
```

These scripts filter interactions, build `inter.txt`, and generate `item_info.jsonline` under `data/<CD/games>/handled/`.

### 2) Prepare the backbone LLM

Create a directory for Qwen2.5-0.5B under this project:

```text
qwen2.5_0.5b/
```

Download the Qwen2.5-0.5B model weights and files into that folder. The provided scripts use this path by default.

### 3) Run LRCL embedding generation

Run from the `ReaEmb/` directory:

`bash experiments/LRCL.bash`

This script trains the LRCL checkpoint and exports item embeddings.

Outputs:

```text
saved/<dataset>/
results/<dataset>/qwen2.5/
```

To switch datasets, edit the `dataset` parameter in `experiments/LRCL.bash`.

### 4) Run CRRL embedding generation

Run from the `ReaEmb/` directory:

`bash experiments/CRRL.bash`

This script loads the LRCL checkpoint, runs CRRL training, and calls `grpo_tuning/eval_noise_grpo.py` to export embeddings.

Outputs:

```text
grpo_tuning/output_grpo/<dataset>/
grpo_tuning/grpo_emb/<dataset>/json/
```

To switch datasets, edit the `dataset` parameter in `experiments/CRRL.bash`.

### 5) Convert embeddings

The `convert.py` scripts convert exported JSONL embeddings into PCA-reduced pickle files.

For CRRL embeddings, run:

```bash
cd grpo_tuning/grpo_emb/<dataset>
python convert.py item_embs_grpo
```

The converted embeddings will be saved under:

```text
grpo_tuning/grpo_emb/<dataset>/handled/
```

`experiments/downstream.bash` will run this conversion automatically if the converted CRRL embedding file is missing.

## Run & Evaluate ReaEmb

### Train & Test Downstream SRS

To train and test the downstream sequential recommender with ReaEmb item embeddings, run:

`bash experiments/downstream.bash`

This script uses the converted CRRL embeddings as the item embedding table and trains the selected SRS backbone. To switch datasets or backbones, edit `dataset` and `model_name` in `experiments/downstream.bash`.

Supported downstream backbones:
- `sasrec_seq`
- `gru4rec`
- `bert4rec`
- `poolrec`

### Outputs

LRCL checkpoints and embeddings:

```text
saved/<dataset>/
results/<dataset>/qwen2.5/
```

CRRL checkpoints and embeddings:

```text
grpo_tuning/output_grpo/<dataset>/
grpo_tuning/grpo_emb/<dataset>/
```

Downstream logs and checkpoints:

```text
downstream/log/<dataset>/
downstream/saved/<dataset>/
```

## Repository Structure

```text
├─ README.md
├─ requirements.txt
├─ main_llm.py                    # LRCL training and embedding export entry
├─ experiments
│  ├─ LRCL.bash                   # Run LRCL training and export
│  ├─ CRRL.bash                   # Run CRRL training and export
│  └─ downstream.bash             # Run downstream SRS training and evaluation
├─ data
│  ├─ yelp/handled/
│  ├─ CD/
│  │  ├─ dataset/                 # Amazon CD preprocessing scripts
│  │  └─ handled/
│  └─ games/
│     ├─ dataset/                 # Amazon Games preprocessing scripts
│     └─ handled/
├─ llm
│  ├─ qwen3.py                    # LLM embedding wrapper
│  ├─ r3_latent_thought.py        # Latent reasoning module
│  ├─ trainer_seq2seq.py          # LRCL trainer
│  ├─ data_processor/             # LRCL data processing and collators
│  └─ peft/                       # Local PEFT components
├─ grpo_tuning
│  ├─ train_noise_grpo.py         # CRRL training entry
│  ├─ eval_noise_grpo.py          # CRRL embedding export
│  ├─ grpo_dataset.py             # Co-occurrence candidate construction
│  ├─ grpo_trainer.py             # CRRL trainer
│  ├─ model.py                    # GRPO model wrapper
│  └─ grpo_emb/                   # CRRL embedding conversion scripts
├─ downstream
│  ├─ main.py                     # Downstream SRS training entry
│  ├─ models/                     # SASRec, GRU4Rec, BERT4Rec, and adapters
│  ├─ generators/                 # Sequential recommendation data loaders
│  ├─ trainers/                   # Downstream training and evaluation
│  └─ utils/                      # Metrics, logging, and arguments
└─ results                        # LRCL embedding conversion scripts
```

## Notes

- The default scripts are configured for 4 GPUs.
- Run LRCL before CRRL, or modify `--lora_path` in `experiments/CRRL.bash` to an existing LRCL checkpoint.
- Run CRRL before downstream training, or modify `llm_emb_file` in `experiments/downstream.bash` to an existing embedding pickle file.
