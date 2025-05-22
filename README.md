# EARnet

**EARnet: Radiographic Reports Generation via Retrieval-Enhanced Cross-Modal Fusion**

This repository contains the official implementation of our paper:
**EARnet: Radiographic Reports Generation via Retrieval-Enhanced Cross-Modal Fusion**

---

## 🔍 Overview

EARnet is a novel framework for automatic radiographic report generation. It addresses key challenges in medical report generation, including cross-modal alignment and diagnostic accuracy, by integrating a retrieval-enhanced fusion mechanism. 

We evaluate EARnet on multiple datasets (IU-Xray, MIMIC-CXR, and a private dental dataset), and it achieves state-of-the-art performance on clinical efficacy metrics.

---


## 🚀 Training & Evaluation

### Train/Evaluate

```bash
python train_full.py 
```

### Result

```bash
python visual.py 
```


---

## 🤝 Acknowledgment

Our project is inspired by and builds upon the following repositories:

* [R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN)
* [Accurate & Fluent Medical X-ray Reports (Xia et al.)](https://github.com/jingjingliang/accurate-xray-report)

We sincerely thank the authors for sharing their excellent work.

---

## 📄 Citation

If you find this project helpful, please consider citing:

```bibtex
@article{yourpaper2024,
  title={EARnet: Radiographic Reports Generation via Retrieval-Enhanced Cross-Modal Fusion},
  author={Your Name and Collaborators},
  journal={...},
  year={2024}
}
```

