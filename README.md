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

---

## Quick Start (Google Colab)

### Prerequisites
- Google account (Colab + Drive)

### 1. Open `main.ipynb` in Colab
```bash
# Option A: Upload notebook file
# File → Upload notebook → select main.ipynb

# Option B: Open directly (replace YOUR_USERNAME)
https://colab.research.google.com/github/YOUR_USERNAME/Estimating-Poses-of-Hands-Interacting-with-Objects/blob/main/main.ipynb
```

### 2. Run All Cells
```bash
# In Colab: Runtime → Run all
```
The single notebook runs the complete workflow:
1. **Setup** - Installs `uv`, clones repo, installs deps, mounts Drive, creates persistent folders, downloads dataset
2. **Data Exploration** - Inspects DexYCB, visualizes batches, verifies active object logic
3. **HaMeR Pretrained Inference** - Evaluates on DexYCB (Table 1 metrics)
4. **HaMeR Fine-tuning** - Trains HaMeR on DexYCB
5. **HOI Training** - Trains full hand-object interaction model
6. **Evaluation** - Computes Table 2 metrics + qualitative visualizations (Figure 4)

### Persistent Storage (Google Drive)
All checkpoints, dataset, and logs persist on Drive:
```
/content/drive/MyDrive/HOI_Project/
├── checkpoints/     # Model weights (persistent!)
├── dataset/         # DexYCB dataset
└── logs/            # TensorBoard logs
```

---

## Configuration

### Main Config: `configs/config.yaml`
All hyperparameters centralized:
- `data.*` - Batch size, augmentations, split ratios
- `hamer.*` - HaMeR architecture
- `hoi.*` - HOI model architecture
- `train.hamer.*` / `train.hoi.*` - Training hyperparameters
- `dexycb_drive_file_id` - Dataset Drive file ID (hardcoded)

**Colab:** Drive paths configured in notebook.

---

## Project Structure

```
├── configs/
│   └── config.yaml          # Single config file
├── checkpoints/             # GITIGNORED - on Google Drive
│   ├── hamer_pretrained/
│   ├── hamer_finetuned/
│   └── hoi_model/
├── dataset/                 # GITIGNORED (except splits/)
│   ├── raw/                 # Dataset from Drive
│   ├── processed/           # Cache
│   └── splits/              # COMMITTED - split indices
├── logs/                    # GITIGNORED
├── main.ipynb               # Single Colab notebook (complete workflow)
├── scripts/
│   └── download_dataset.py  # Download from Drive
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
- **Always save to Google Drive**: `/content/drive/MyDrive/HOI_Project/checkpoints/`
- Config `checkpoint_dir` points to Drive
- `save_every: 5` in config → checkpoint every 5 epochs
- **Auto-resume** on next session

### 12h Timeout - Keep Alive
To prevent disconnect after 12h inactivity:
```javascript
// Browser console (F12) → paste:
setInterval(() => {
  const btn = document.querySelector('#connect-button');
  if (btn) btn.click();
  console.log('Keep-alive ping');
}, 5 * 60 * 1000);
```
Or install Chrome extension **"Colab Keep Alive"**.

### GPU Variability
Colab free tier: T4 / K80 / P100 (random assignment).
- `batch_size: 32` in config (reduce if OOM)
- `mixed_precision: true` + `grad_accum_steps: 2` for stability
- `num_workers: 0` required (multiprocessing doesn't work)

---

## Experiment Tracking: TensorBoard (Built-in)

**Why TensorBoard:** Zero setup, built into PyTorch, works in Colab.

**Usage:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=config.log_dir)
writer.add_scalar("Loss/train", loss, epoch)
writer.add_scalar("MPJPE/val", mpjpe, epoch)
writer.add_figure("Predictions", fig, epoch)
writer.close()
```

**In Colab:** Load TensorBoard extension:
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/HOI_Project/logs
```

**What to Log:**
- Train/val loss per epoch
- All Table 1 & 2 metrics (MPJPE, MPVPE, ADD, AUC, RR, PA)
- Qualitative visualizations (overlay predictions)

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
| `pytorch3d` | Chamfer distance, mesh ops |

**Dependency Management:** Only `pyproject.toml` + `uv.lock`. No `requirements.txt`.

---

## Development Notes

### Architecture: `src/` + `main.ipynb`
- `src/` = **reusable package** (logic, models, metrics, utils)
- `main.ipynb` = **single entry point** (complete workflow)

### Active Object Definition
> "An active object is strictly defined as an object in direct contact with the hand in a given frame. Frames without active contact must be identified and handled accordingly."

Implement in `src/data/dataset.py` → `_is_active_object()`.

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
- **HaMeR Paper** [9]: Reconstructing hands in 3d with transformers (CVPR 2024)
- **DexYCB Dataset** [2]: https://dex-ycb.github.io/
- **Dataset Drive Link**: https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk
- **MANO Model** [11]: Embodied hands (requires registration)
- **YCB Objects** [1]: https://www.ycbbenchmarks.com/

---

## License
Academic use only - Politecnico di Torino, Advanced ML 2026-27.