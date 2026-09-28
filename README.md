# CueMamba: Liver Tumor Segmentation with Dual-Branch Refinement

This repository provides the official implementation of **CueMamba**, a 3D medical image segmentation architecture that integrates dual-branch refinement into the U-Mamba backbone for liver tumor segmentation on arterial-phase contrast-enhanced MRI.

## Architecture

CueMamba extends U-Mamba with a **Dual-Branch Gating Module (DBGM)** that addresses two key challenges in liver tumor segmentation:

- **Boundary Ambiguity**: The local-background difference branch derives boundary-sensitive information by comparing each feature with its local average.
- **Tumor Heterogeneity**: The prototype similarity branch provides foreground-consistency information from confidence-weighted features.

A gated fusion mechanism measures the discrepancy between the two branch responses and refines the original representation through residual fusion.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install nnU-Net (required for training pipeline)
pip install nnunetv2

# Install mamba-ssm
pip install mamba-ssm causal-conv1d
```

## Project Structure

```
CueMamba/
├── models/
│   ├── dual_branch_gating.py          # Dual-Branch Gating Module (DBGM)
│   ├── CueMamba.py                    # CueMamba model (MambaLayer + DBGM)
│   └── CueMambaEnc.py                 # Encoder variant with DBGM
├── trainers/
│   └── nnUNetTrainerCueMamba.py       # nnU-Net trainer with DBGM
├── scripts/
│   ├── train.sh                       # Training launch script
│   └── predict.sh                     # Inference script
├── evaluation/
│   └── *.py                           # Evaluation metrics (Dice, NSD, etc.)
├── results/
│   ├── exp1.drawio.pdf                # Dice score comparison
│   └── exp2.drawio.pdf                # Efficiency comparison
├── example_usage.py                   # Usage example
├── train_standalone.py                # Standalone training script
└── requirements.txt
```

## Usage

### Training

```bash
# Via nnU-Net pipeline
nnUNetv2_train Dataset501_HCC_MRI 3d_fullres 0 -tr nnUNetTrainerUMambaBotPlugin

# Via standalone script
python train_standalone.py --data_dir /path/to/data --output_dir /path/to/output
```

### Inference

```bash
python example_usage.py
```

## Results

CueMamba achieves a Dice coefficient of **0.6061** on the liver tumor segmentation benchmark, outperforming U-Mamba (0.6047), Swin-UMamba+ (0.6034), UNETR (0.4450), and Swin-UNETR (0.3140), while retaining the linear sequence complexity of the U-Mamba backbone.

## Citation

If you find this work useful, please cite:

```bibtex
@article{cuemamba2024,
  title={CueMamba: Liver Tumor Segmentation with Dual-Branch Refinement},
  author={},
  year={2024}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
