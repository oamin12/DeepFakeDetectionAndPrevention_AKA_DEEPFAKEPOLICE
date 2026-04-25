# Deepfake Detection Models Summary

Snapshot date for popularity: **2026-03-23**.

> **Note on popularity:** this column uses a single compact public signal per item to keep the table aligned. In most cases it is **GitHub stars**. For the Hugging Face model, I keep the **HF monthly downloads** in the same cell.

## Main comparison table

| Model / Approach | Face-specific or general? | Image or video? | Approach family | Why it falls under that approach | Released / published | Paper | Code / model / weights | Popularity snapshot |
|---|---|---|---|---|---|---|---|---|
| **MLEP** | **General** AI-generated / deepfake images | Image | **CNN-based** | The paper says MLEP can be used to build a robust **CNN-based classifier**, and the repo uses a **ResNet-50** training setup. | **2025-09-18** (OpenReview / NeurIPS 2025 listing) | https://openreview.net/forum?id=Bsska2ayiy | GitHub: https://github.com/fkeufss/MLEP | GitHub: **4★** |
| **Defocus-Deepfake-Detection** | **General**, especially relevant to **facial manipulation** | Image | **CNN backbone (Xception)** | The method uses **defocus blur maps** as forensic cues, and the official repo trains/tests with **legacy Xception** and provides pretrained weights. | **2025-09-27** (arXiv) / CIKM 2025 | https://arxiv.org/abs/2509.23289 | GitHub: https://github.com/irissun9602/Defocus-Deepfake-Detection | GitHub: **3★** |
| **mne2darsh n3mlo download - Bi-Level Optimization (BLADES / AIGFD_BLO)** | **Face-specific** | Image | **CNN backbone + lightweight detector head** | The method is explicitly for **AI-generated face detection**. It uses a pretrained **ResNet-50** face encoder and then detects fake faces with a **GMM** or a **2-layer MLP**. | **2025-07-30** (arXiv) / ICCV 2025 | https://arxiv.org/abs/2507.22824 | GitHub: https://github.com/MZMMSEC/AIGFD_BLO | GitHub: **4★** |
| **done 50% - CommunityForensics-DeepfakeDet-ViT** | **General** AI-generated / fake images | Image | **Vision Transformer (ViT)** | The Hugging Face model card states this is a **ViT-Small** model finetuned for fake-image detection. | **2024-11-06** (paper on HF page) | https://huggingface.co/papers/2411.04125 | Hugging Face: https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT ; GitHub: https://github.com/JeongsooP/Community-Forensics | HF: **623,254 downloads/month** ; GitHub: **32★** |
| **done 70% - M2F2-Det** | **Face-specific** | Image / face forgery analysis | **Vision-language / multimodal (CLIP + LLM)** | The paper explicitly introduces a **multi-modal face forgery detector** using **CLIP** for detection/generalization and an **LLM** for textual explanations. | **2025-03-26** (arXiv) / CVPR 2025 Oral | https://arxiv.org/abs/2503.20188 | GitHub: https://github.com/CHELSEA234/M2F2_Det ; HF weights in repo: https://huggingface.co/CHELSEA234/llava-v1.5-7b-M2F2-Det | GitHub: **106★** |
| **DFD-FCG** | **Face-specific** | **Video** | **Foundation-model adaptation** | The method is for **video-based deepfake detection** and uses **facial component / feature guided adaptation** for a foundation model. It is face-specific, but not a still-image model. | **2025** (CVPR 2025) | Repo citation / project information in repo | GitHub: https://github.com/aiiu-lab/DFD-FCG | GitHub: **49★** |
| **FSFM** | **Face-specific** | Image | **ViT-based foundation model** | FSFM learns facial representations with self-supervised pretraining and uses a **ViT-B/16** backbone for transferable face-security tasks including deepfake detection. | **2025** (CVPR 2025) | CVPR paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_FSFM_A_Generalizable_Face_Security_Foundation_Model_via_Self-Supervised_Facial_CVPR_2025_paper.pdf | GitHub: https://github.com/wolo-wolo/FSFM-CVPR25 | GitHub: **115★** |
| **Forensics Adapter** | **Face-specific** | Image | **CLIP / vision-language adapter** | The method adapts **CLIP** to face forgery detection using a lightweight adapter that focuses on blending boundaries and other forgery-specific cues. | **2025** (CVPR 2025) | CVPR paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.pdf | GitHub: https://github.com/OUC-VAS/ForensicsAdapter | GitHub: **90★** |

## Additional popular recent multimodal LLM / MLLM options

These are not added to the main comparison table above, but they are strong candidates if you want to compare against **explainable multimodal** detectors.

| Model | Face-specific or general? | Image or video? | Approach family | Released / published | Paper | Code / model / weights | Popularity snapshot |
|---|---|---|---|---|---|---|---|
| **AIGI-Holmes** | **General** AI-generated images | Image | **MLLM / multimodal LLM** | **2025-07-03** (arXiv) / ICCV 2025 | https://arxiv.org/abs/2507.02664 | GitHub: https://github.com/wyczzy/AIGI-Holmes | GitHub: **167★** |
| **FakeVLM** | **General**, supports both synthetic image and **DeepFake** detection | Image | **MLLM / multimodal LLM** | **2025-03-19** (arXiv) / NeurIPS 2025 | https://arxiv.org/abs/2503.14905 | GitHub: https://github.com/opendatalab/FakeVLM ; HF model: https://huggingface.co/lingcco/fakeVLM | GitHub: **128★** ; HF: **357 downloads/month** |
| **LEGION** | **General** synthetic image detection | Image | **MLLM / multimodal large language model** | **2025-03-19** (arXiv) / ICCV 2025 Highlight | https://arxiv.org/abs/2503.15264 | GitHub: https://github.com/opendatalab/LEGION | GitHub: **75★** |

## Quick notes

- **Face-specific models in the main table:** **BLADES**, **M2F2-Det**, **DFD-FCG**, **FSFM**, **Forensics Adapter**
- **General image detectors in the main table:** **MLEP**, **Defocus-Deepfake-Detection**, **Community Forensics**
- **Transformer / ViT-style approaches:** **Community Forensics**, **FSFM**
- **Vision-language approaches:** **M2F2-Det**, **Forensics Adapter**
- **MLLM / multimodal LLM options:** **AIGI-Holmes**, **FakeVLM**, **LEGION**
- **Video-based rather than image-based:** **DFD-FCG**
- **Older than your late-2025/2026 preference:** **Community Forensics** is from **2024**, but it is included because you explicitly asked for it
