# RAFM: Retrieval-Augmented Flow Matching for Unpaired CBCT-to-CT Translation

Official research codebase for **“RAFM: Retrieval-Augmented Flow Matching for Unpaired CBCT-to-CT Translation”**.

This repository provides:
- Training, testing/inference, and evaluation pipelines
- A retrieval-augmented component (Memory Bank + foundation feature extractor)
- Utilities for saving results and computing metrics

---

## Repository Layout

```
RAFM/
  main.py                  # Entry: optionally runs train -> test -> evaluation
  train.py                 # Training
  test.py                  # Testing / inference (saves synthesized results)
  evaluation.py            # Evaluation (optional reconstruction/segmentation/metrics)
  requirements.txt

  file_config/             # JSON configs
    base.json
    train.json
    test.json
    evaluation.json
    experiments.json
    experiments/
      RAFM.json

  code_config/             # Config parsing/merging
  code_dataset/            # Dataset definitions (aligned/unaligned 2D)
  code_model/              # Model definitions (e.g., i2irf)
  code_network/            # Networks/tools (RF/UNet/DINOv3 adapter, Memory Bank)
  code_record/             # Logging/visualization
  code_util/               # IO, preprocessing, reconstruction, metrics, etc.
```

---

## Installation

Recommended:
- Python 3.10+
- PyTorch (CUDA optional, depending on your environment)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Configs are under `file_config/`:
- Base config: `file_config/base.json`
- Stage configs: `file_config/train.json`, `file_config/test.json`, `file_config/evaluation.json`
- Experiment switch/index: `file_config/experiments.json`
- Experiment definition (default): `file_config/experiments/RAFM.json`

Typical edits:
1) Set dataset root/path in `file_config/base.json` (e.g., `dataset.dataroot`).
2) Select dataset mode (aligned/unaligned) in `base.json` / stage configs.
3) Tune RAFM/model parameters in `file_config/experiments/RAFM.json`.

---

## Data

By default, the dataset root is configured via:
- `file_config/base.json -> dataset.dataroot` (default `./file_dataset`)

Dataset implementations:
- `code_dataset/aligned2D_dataset.py`
- `code_dataset/unaligned2D_dataset.py`

### Suggested dataset layout

Place your data under `dataset.dataroot` (default: `./file_dataset`). The project supports both **aligned (paired slice-to-slice)** and **unaligned (unpaired)** 2D settings.

**A) Unaligned 2D (recommended for unpaired CBCT↔CT)**

Example structure:

```
file_dataset/
  trainA/          # domain A, e.g., CBCT slices
    xxx_0001.nii.gz
    xxx_0002.nii.gz
  trainB/          # domain B, e.g., CT slices
    yyy_0001.nii.gz
    yyy_0002.nii.gz
  testA/
    ...
  testB/
    ...
```

- `A` and `B` are two domains (e.g., A=CBCT, B=CT).
- In unaligned mode, file names do **not** need to match between domains.

**B) Aligned 2D (paired / aligned)**

Example structure:

```
file_dataset/
  trainA/
    case001_0001.nii.gz
    case001_0002.nii.gz
  trainB/
    case001_0001.nii.gz
    case001_0002.nii.gz
  testA/
    ...
  testB/
    ...
```

- In aligned mode, samples are paired by **sorted order and/or matching file names** (depending on the dataset implementation). To avoid ambiguity, keep **the same file names** in `A` and `B` for each pair.

> Notes:
> - The expected folders (`trainA/trainB/testA/testB`) and file extensions are controlled by your JSON configs (e.g., `dataset.data_format`) and dataset code under `code_dataset/`.
> - If your data is 3D volumes, you typically need to preprocess/slice them into 2D samples before training, or implement a 3D dataset.

---

## Usage

### Run the full pipeline

`main.py` loads configs and runs (optionally) **train → test → evaluation**.

```bash
python main.py
```

You can enable/disable stages by editing the flags in `main.py`:
- `do_train`
- `do_test`
- `do_eval`

### Run individual stages

```bash
python train.py
python test.py
python evaluation.py
```

---

## Outputs

Paths are controlled by JSON configs (see `file_config/*`). Common defaults:
- Records/logs: `file_config/base.json -> record.record_dir` (default `./file_record`)
- Inference results: `file_config/test.json -> result.result_dir` (default `./file_result`)

Actual `work_dir` naming/assembly is handled by the config parser (see `code_config/parser.py`).

---

## Citation

If you find this repository useful, please cite the RAFM paper

<!-- ```bibtex
@article{RAFM,
  title   = {Retrieval-Augmented Flow Matching for Unpaired CBCT-to-CT Translation},
  author  = {…},
  journal = {…},
  year    = {…}
}
``` -->

---

## License / Disclaimer

- Research use only.
- For any third-party components (e.g., DINOv3), please follow their respective licenses.
- Outputs from medical imaging models must not be used for clinical diagnosis. Users are responsible for compliance and risk management.
