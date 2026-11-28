# HOI Pose Estimation - PoliTo Advanced ML 2026-27

## Overview
Hand-Object Interaction (HOI) pose estimation with Transformers (HaMeR) on DexYCB dataset.

**Project Goals (3 mandatory parts):**
1. **Data Inspection** - Dataset exploration, active object detection
2. **HaMeR on DexYCB** - Pretrained inference + fine-tuning (Table 1)
3. **HOI Model** - HaMeR extension for hand+object (Table 2 + Figure 4)

**Optional Extension (choose one):**
- Contact Pose Estimation (ContactNet)
- Multi-view Pose Estimation
- Knowledge Distillation

---

## Team
- Michelangelo Ungolo (s349109)
- Matteo Gaffoglio (s......)
- ...

---

## How we work

| What | Where | Notes |
|------|-------|-------|
| Code | **GitHub** (this repo) | Everyone has their own clone and works on their own branch, then opens a pull request to `main` |
| Data, MANO, checkpoints, outputs | **Your own Google Drive**, `MyDrive/HOI_Project` | Created and filled by `main.ipynb`; each member has their own copy |
| GPU runs | **Google Colab** | Only `main.ipynb`; Colab wipes its disk at every disconnection |
| Metrics and curves | **Weights & Biases** | One run per experiment, visible to the whole team |

Workflow:
1. Edit the `.py` files on your PC (e.g. VS Code), commit and push to your branch.
2. On Colab, set `BRANCH` in cell 1.1 and re-run it (`git pull`).
3. Run the sections you need. Checkpoints and outputs go to your Drive, metrics to W&B.

Never commit the dataset, `MANO_*.pkl`, checkpoints or W&B files (see `.gitignore`).

---

## Quick Start (Google Colab)

### Prerequisites (once per member)
- Google account (Colab + Drive) with ~16 GB free on Drive (dataset ~13.9 GB + checkpoints)
- MANO model: register at https://mano.is.tue.mpg.de, download *Models & Code* and upload
  `mano_v1_2/models/MANO_RIGHT.pkl` and `MANO_LEFT.pkl` into `MyDrive/HOI_Project/mano/`
- W&B account: copy your API key from https://wandb.ai/authorize into the Colab Secrets
  (key icon on the left) as `WANDB_API_KEY`, with notebook access enabled

### Run
1. Open `main.ipynb` in Colab: File -> Open notebook -> GitHub -> paste the repo URL.
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Run the **Setup** section (1.1 -> 1.6):
   - clones/updates the repo and installs the dependencies of `pyproject.toml`
   - mounts **your** Drive and creates `MyDrive/HOI_Project`
   - **first run ever:** downloads DexYCB subject-01, calibration and YCB models (~13.9 GB) into your Drive
   - **every session:** copies the archives to the Colab disk and extracts them in `/content/dexycb` (~10-15 min)
   - logs in to W&B
4. Run the sections you need (2 -> 6).

### Your Google Drive
```
MyDrive/HOI_Project/
├── dataset/       # 20200709-subject-01.tar.gz, calibration.tar.gz, models.tar.gz
├── mano/          # MANO_RIGHT.pkl, MANO_LEFT.pkl (licence: never on GitHub)
├── checkpoints/   # one sub-folder per model: hamer_pretrained/, hamer_finetuned/, hoi_model/
├── logs/
└── outputs/       # figures and tables for the report
```

### Local data on Colab (`/content/dexycb`, rebuilt every session)
```
20200709-subject-01/<sequence>/meta.yml                       # objects, grasped object, hand side
20200709-subject-01/<sequence>/<camera serial>/color_XXXXXX.jpg
                                              aligned_depth_to_color_XXXXXX.png
                                              labels_XXXXXX.npz   # seg, pose_y, pose_m, joint_3d, joint_2d
calibration/                                                   # intrinsics, extrinsics, MANO betas
models/                                                        # YCB object meshes
```
100 sequences x 8 cameras, ~58k frames. Units in the labels are **meters** (metrics in mm).

---

## Configuration

### Main Config: `configs/config.yaml`
All hyperparameters and paths centralized:
- `drive_root`, `drive_dataset_dir`, `mano_dir`, `checkpoint_dir`, `log_dir`, `output_dir` - your Drive
- `data_root` - local extraction folder on the Colab disk
- `dexycb.files` - official DexYCB Google Drive file IDs (downloaded by `scripts/download_dataset.py`)
- `wandb.*` - W&B project / entity / mode
- `data.*` - Batch size, augmentations, split ratios
- `hamer.*` - HaMeR architecture
- `hoi.*` - HOI model architecture
- `train.hamer.*` / `train.hoi.*` - Training hyperparameters, `checkpoint_subdir` per model

---

## Project Structure

```
├── configs/
│   └── config.yaml          # Single config file
├── dataset/
│   └── splits/              # COMMITTED - split indices
├── main.ipynb               # Single Colab notebook (complete workflow)
├── scripts/
│   └── download_dataset.py  # Drive download + local extraction + MANO check
├── src/                     # Python package (importable)
│   ├── data/                # Dataset, transforms, splits
│   ├── models/              # HaMeR, HOI, losses
│   ├── utils/               # Metrics, geometry, viz, checkpoint
│   └── training/            # Trainer, callbacks
├── tests/                   # Minimal tests
├── .gitignore
├── .python-version          # 3.10
├── README.md
├── pyproject.toml           # Dependencies + tool config
└── uv.lock                  # Lockfile (COMMITTED)
```

---

## Key Colab Points

### Persistent Checkpoints (Required)
- Checkpoints are saved on **your** Drive: `MyDrive/HOI_Project/checkpoints/<checkpoint_subdir>/`
- Each model has its own sub-folder (`train.*.checkpoint_subdir`), so a run never resumes from another model
- `save_every: 5` -> checkpoint every 5 epochs, auto-resume on the next session

### 12h Timeout - Keep Alive
To prevent disconnect after 12h inactivity:
```javascript
// Browser console (F12) -> paste:
setInterval(() => {
  const btn = document.querySelector('#connect-button');
  if (btn) btn.click();
  console.log('Keep-alive ping');
}, 5 * 60 * 1000);
```
Or install Chrome extension **"Colab Keep Alive"**.

### GPU Variability
Colab free tier: T4 (random assignment, availability not guaranteed).
- `batch_size: 32` in config (reduce if OOM)
- `mixed_precision: true` + `grad_accum_steps: 2` for stability
- `num_workers: 0` required (multiprocessing doesn't work)

---

## Experiment Tracking: Weights & Biases

**Why W&B:** used in the course labs, one shared place for the runs of every team member.

**Usage:**
```python
import wandb

run = wandb.init(project=config.wandb.project, entity=config.wandb.entity,
                 mode=config.wandb.mode, name="hoi-lr5e-5-bs32", config=dict(config.train.hoi))
wandb.log({"train/loss": loss, "val/mpjpe": mpjpe}, step=epoch)
wandb.log({"predictions": wandb.Image(fig)})
run.finish()
```

**What to Log:**
- Train/val loss per epoch
- All Table 1 & 2 metrics (MPJPE, MPVPE, ADD, AUC, RR, PA)
- Qualitative visualizations (overlay predictions)

Name runs clearly (`<model>-<main hyperparameters>`) and set `wandb.entity` to the team name.

---

## Testing & Code Quality

```bash
# Tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/
```

**Config in `pyproject.toml`:** Ruff (line-length=100), MyPy (py310, relaxed for ML).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning |
| `numpy`, `scipy` | Scientific computing |
| `opencv-python-headless` | Image processing |
| `pillow` | Image I/O |
| `omegaconf` | Configuration |
| `tqdm` | Progress bars |
| `matplotlib` | Plotting |
| `trimesh`, `pyrender` | **Mesh rendering for Figure 4** |
| `gdown` | Google Drive download |
| `smplx` | MANO model |
| `timm` | ViT backbone |
| `wandb` | Experiment tracking |
| `pytorch3d` | Chamfer distance, mesh ops (optional) |

**Dependency Management:** Only `pyproject.toml` + `uv.lock`. No `requirements.txt`.
After changing the dependencies run `uv lock` and commit the updated `uv.lock`.
On Colab: `uv pip install --system -r pyproject.toml` (cell 1.2).

---

## Development Notes

### Architecture: `src/` + `main.ipynb`
- `src/` = **reusable package** (logic, models, metrics, utils)
- `main.ipynb` = **single entry point** (complete workflow)

### Active Object Definition
> "An active object is strictly defined as an object in direct contact with the hand in a given frame. Frames without active contact must be identified and handled accordingly."

Implement in `src/data/dataset.py` -> `_is_active_object()`.
`meta.yml` gives the grasped object of each sequence (`ycb_ids[ycb_grasp_ind]`); contact in each frame still has to be computed.

### Target Metrics (Table 2 Baseline)
| Metric | Baseline | Target |
|--------|----------|--------|
| MPVPE | 72.82 mm | < 72.82 |
| PAV | 8.80 mm | < 8.80 |
| MPJPE | 57.62 mm | < 57.62 |
| PAJ | 11.02 mm | < 11.02 |
| ADD | 96.90 mm | < 96.90 |
| PD | 2.42 mm | < 2.42 |

---

## References
- **HaMeR Paper** [9]: Reconstructing hands in 3d with transformers (CVPR 2024) - https://github.com/geopavlakos/hamer
- **DexYCB Dataset** [2]: https://dex-ycb.github.io/
- **Dataset Drive Link**: https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk
- **MANO Model** [11]: Embodied hands (requires registration) - https://mano.is.tue.mpg.de
- **YCB Objects** [1]: https://www.ycbbenchmarks.com/

---

## License
Academic use only - Politecnico di Torino, Advanced ML 2026-27.
