# AI-Driven Generation of DprE1 Inhibitors

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project presents a deep reinforcement learning pipeline for *de novo* molecular design of **DprE1 (Decaprenylphosphoryl-β-D-ribose 2'-epimerase) inhibitors** using a hybrid reward model combining Graph Neural Networks (GNN) and gradient boosting, fine-tuned via Proximal Policy Optimization (PPO).

DprE1 is a critical enzyme in *Mycobacterium tuberculosis* cell wall biosynthesis, making it an attractive target for novel anti-tuberculosis drug development.

## Key Features

- **SELFIES-based molecular representation** for 100% validity guarantee
- **Hybrid reward model** combining GNN embeddings and Morgan fingerprints
- **PPO fine-tuning** for optimized molecule generation
- **Multi-objective optimization** balancing biological activity (80%) and drug-likeness (20%)
- **Automated drug-likeness assessment** using Lipinski's Rule of Five and QED scores

## Methodology

### Pipeline Overview

```
ChEMBL Dataset → Filtering → SELFIES Encoding → RNN Pretraining
                                                        ↓
DprE1 Activity Data → GNN + XGBoost Hybrid Model → Reward Function
                                                        ↓
                                        PPO Fine-tuning → Novel Molecules
```

### Dataset

- **ChEMBL v35**: 2,854,815 molecules (filtered to ~500,000 drug-like molecules)
- **DprE1 Activity Dataset**: 1,520 molecules with binary activity labels
- **Filters Applied**: SMILES length (6-150), heavy atoms (5-50), valid RDKit molecules

### Model Architecture

1. **Prior Agent**: 2-layer LSTM (512 hidden units, 128 embedding dimension)
2. **GNN Embedder**: Graph Isomorphism Network (2 → 128 → 128 dimensions)
3. **Hybrid Classifier**: GNN embeddings (128D) + Morgan fingerprints (2048D) → XGBoost
4. **Reward Function**: R = 0.8 × P(Active) + 0.2 × QED

### Training Details

- **Pretraining**: 5 epochs on 100K-500k ChEMBL molecules
- **Hybrid Model**: 20 epochs on DprE1 dataset (80/20 split)
- **PPO Fine-tuning**: 200 iterations, batch size 64, KL coefficient 0.05

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 12-16 GB RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/TridibDatta/dpre1-inhibitor-discovery.git
cd dpre1-inhibitor-discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download ChEMBL Data and Pretrain RNN

```bash
python src/download_chembl_big_data_and_pretraining.py
```

This script will:
- Download ChEMBL v35 dataset
- Filter molecules based on drug-likeness criteria
- Convert SMILES to SELFIES
- Pretrain the LSTM language model

### 2. Train Hybrid Reward Model

```bash
python src/hybrid_model.py
```

This script will:
- Load DprE1 activity dataset
- Train GNN embedder
- Generate molecular fingerprints
- Train XGBoost classifier
- Save trained models

### 3. Fine-tune with PPO

```bash
python src/finetune_ppo.py
```

This script will:
- Load pretrained RNN and hybrid model
- Initialize PPO agent
- Generate molecules over 200 iterations
- Save top-scoring candidates

### 4. Analyze Results

Results are saved in `results/` directory:
- `generated_molecules.csv`: Top-scoring molecules with SMILES, scores, and properties
- `training_curves.png`: Reward progression during PPO training
- `property_distributions.png`: Molecular property distributions

## Project Structure

```
dpre1-inhibitor-discovery/
├── data/
│   └── cleaned_dpre1_data.csv          # DprE1 activity dataset
├── src/
│   ├── download_chembl_big_data_and_pretraining.py  # Data prep & pretraining
│   ├── hybrid_model.py                  # GNN + XGBoost training
│   ├── finetune_ppo.py                  # PPO fine-tuning
│   ├── models.py                        # Neural network architectures
│   └── utils_featurization.py           # Molecular featurization utilities
├── results/
│   └── (generated outputs)
├── docs/
│   └── methodology.md                   # Detailed methodology
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                          # Git ignore rules
```

## Results

### Performance Metrics

- **Validity**: >95% (SELFIES guarantee)
- **Uniqueness**: ~85-90%
- **Novelty**: ~70-80% (Tanimoto < 0.85 to training set)
- **Lipinski Pass Rate**: 60-70%
- **Mean QED Score**: 0.5-0.7
- **Hybrid Model Accuracy**: 0.85-0.90
- **Hybrid Model AUC-ROC**: 0.85-0.92

### Generated Molecules

The pipeline typically generates 50-100 high-scoring candidates with:
- DprE1 activity probability > 0.7
- QED score > 0.5
- Lipinski violations ≤ 1

## Computational Requirements

- **Hardware**: NVIDIA Tesla T4/V100 GPU (or equivalent)
- **Training Time**:
  - Pretraining: ~15-20 minutes
  - Hybrid model: ~5-10 minutes
  - PPO fine-tuning: ~30-45 minutes

## Citation

This project is under JCIM review 
```

## References

1. ChEMBL Database: https://www.ebi.ac.uk/chembl/
2. SELFIES: Krenn et al., *Machine Learning: Science and Technology* (2020)
3. Proximal Policy Optimization: Schulman et al., arXiv:1707.06347 (2017)
4. Graph Isomorphism Networks: Xu et al., ICLR (2019)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ChEMBL database team for molecular data
- RDKit community for cheminformatics tools
- PyTorch Geometric developers
- Google Colab for computational resources

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This is a research project. Generated molecules require experimental validation before any therapeutic use.
