# GFRestimation

GFRestimation is a PyTorch-based framework for estimating split renal glomerular filtration rate (GFR) from contrast-enhanced abdominal CT.

The pipeline has two main parts:

- **Stage Model** (`Deep Learning Model/Stage Model/`): per-phase 3D UNet models (arterial / venous) that learn CT representations and estimate bilateral GFR.
- **Combined Model** (`Deep Learning Model/Combined Model/`): fuses arterial and venous CT features with clinical variables and preliminary predictions to output final left/right GFR.

## Code Layout

- `Deep Learning Model/Stage Model/`
  - `dataloader.py` – Load and preprocess NIfTI CT volumes and labels into PyTorch `DataLoader`s.
  - `train.py` / `evaluate.py` – Train and evaluate a single-phase 3D UNet defined in `Models/UNet_Model*.py`, configured by `parameter.json`.
- `Deep Learning Model/Combined Model/`
  - `UNet.py` – Defines 3D UNet backbones and the combined model `mlp` that loads pretrained arterial/venous models and predicts bilateral GFR.
  - `dataloader.py` – Multimodal loader combining arterial CT, venous CT, clinical/radiomic features, and preliminary prediction labels.
  - `train.py` / `evaluate.py` – Train and evaluate the combined model, saving checkpoints and basic metrics/plots under `model/all/`.

## Requirements

- Python, PyTorch, and common scientific packages including `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `nibabel`, `tqdm`, `scikit-image`.
- 3D CT volumes in NIfTI format and corresponding CSV labels prepared under `data_All/<hospital>/` (not included).

## nnUnet

The nnU-Net parameters used in our experiments can be accessed via the following URL: <https://huggingface.co/cnWxy/nnUnet/tree/main>.

