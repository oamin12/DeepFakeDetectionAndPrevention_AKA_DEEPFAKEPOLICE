# Amin's Models - MLEP & FSFM

Inference only — no training or fine-tuning. Using pretrained weights as-is to evaluate on our datasets.

## Models

### MLEP (General AI-Generated Image Detector)
- **Paper**: [MLEP - NeurIPS 2025](https://openreview.net/forum?id=Bsska2ayiy)
- **Architecture**: Modified ResNet-50 with Multi-scale Local Entropy Preprocessing
- **Type**: General (not face-specific)
- **Output**: sigmoid > 0.5 = fake
- **Weights**: Pretrained checkpoint from the authors (no retraining)

### FSFM (Face Security Foundation Model)
- **Paper**: [FSFM - CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_FSFM_A_Generalizable_Face_Security_Foundation_Model_via_Self-Supervised_Facial_CVPR_2025_paper.pdf)
- **Architecture**: ViT-B/16
- **Type**: Face-specific
- **Output**: softmax[:,1] > 0.5 = fake
- **Weights**: Pretrained checkpoint fine-tuned on FF++ c23 (downloaded from HuggingFace, used as-is)

## Setup (after cloning this repo)

```bash
# 1. Clone model repos
git clone https://github.com/fkeufss/MLEP.git
git clone https://github.com/wolo-wolo/FSFM-CVPR25.git

# 2. Install dependencies
pip install torch torchvision timm tqdm pillow huggingface_hub

# 3. Download FSFM weights (MLEP weights come with its repo)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(local_dir='fsfm_weights', repo_id='Wolowolo/fsfm-3c',
                filename='finetuned_models/FF++_c23_32frames/checkpoint-min_val_loss.pth')
"
```

## Running Evaluation

```python
from mlep_fast_evaluator import load_mlep_model, evaluate_all_datasets_mlep_fast
from fsfm_fast_evaluator import load_fsfm_model, evaluate_all_datasets_fsfm_fast

# MLEP
model, device = load_mlep_model()
results_mlep = evaluate_all_datasets_mlep_fast(model, device)

# FSFM
model, device = load_fsfm_model(download_from_hf=False)
results_fsfm = evaluate_all_datasets_fsfm_fast(model, device)
```

## Output

Results saved to `Results/MLEP_ResNet50_{dataset}.json` and `Results/FSFM_ViT_B16_{dataset}.json`.

## Files to Push

| File | Purpose |
|------|---------|
| `mlep_fast_evaluator.py` | MLEP evaluation script |
| `fsfm_fast_evaluator.py` | FSFM evaluation script |
| `README_Amin.md` | This file |
