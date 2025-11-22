# Predictive vision-language integration in the human visual cortex

---
## About
This repo contains code for analyzing the predictive vision-language integraion on human fMRI BOLD signals.

---
## Requirements

This project requires the following software and environment setup.

### **Operating System and Hardware**
- OS: CentOS 7.6.1810  
- GPU Driver: 535.129.03  
- CUDA: 12.2
- GPU: Single NVIDIA A100 80GB recommended  
  - Note: Running on 40GB GPUs may cause out-of-memory errors for some participants.

### **Python Environment**
We provide a full `requirements.txt` containing all Python dependencies.  
To reproduce the environment exactly, we recommend creating a clean **Python 3.8** environment via Conda:

```bash
conda create -n vli_env python=3.8.17
conda activate vli_env
pip install -r requirements.txt
```

### Neuroimaging Dependencies

This project requires the following neuroimaging software.  
Please install each tool following the official documentation provided on their websites.

| Software       | Required Version | Website |
|----------------|------------------|---------|
| **FSL**        | 6.0              | https://fsl.fmrib.ox.ac.uk |
| **AFNI**       | 21.3.11          | https://afni.nimh.nih.gov |
| **FreeSurfer** | 7.4.0            | https://surfer.nmr.mgh.harvard.edu |

**Typical installation time**  
- Python environment (via `conda` + `pip install -r requirements.txt`): **within ~1 hour** under standard network conditions.  
- FSL / AFNI / FreeSurfer installations vary by OS and download speed; allow **1~3 hours** for complete installation and configuration of all neuroimaging tools.

---

## Directory Overview

The main source files are the following:


`./Paper_figure.ipynb` contains the code for plotting the figure shown in the paper. Also, the jupyter notebook contains code for neural manifold analysis (in Sec. Figure 6). The original data used to reproduce the results in this notebook can be found at https://osf.io/gj3qx/

`./code/` contains scripts for analyzing fMRI data.

* `extract_roi.py` contains the code for extract several ROIs of each participant, including suppressed/enhanced region, early visual cortex (EVC), V1-V3 separately, suppressed/enhanced EVC, suppressed/enhanced inferior frontal sulcus (IFS), middle frontal gyrus (MFG), etc.
* `train_encode.py` contains the code for training the caption-evoked voxel-wise encoding model.
* `semantic_representation_learning.py` contains the code for training the BrainCLIP model to align the caption response and image response into the same embedding space.
* `calculate_dissimilarity_correlation.py` contains the code for calculating the correlation between prediction error defined by features' consine distance and BOLD's beta activation.
* `semantic_cluster.py` contains the code for unsupervised clustering of stimuli using CLIP features.
* `visualization_SRL.py` contains the code for visualizing the raw brain response space, stimuli CLIP embedding space and BrainCLIP-encoded embedding space.
* `demo.py` contains example code demonstrating the use of the models in the `./code/` folder using a simulated dataset. 

`./model_cache/` is the directory used for storing cached models required for running the demos.

- `download_model.sh` contains example commands for downloading the pretrained models.

Before running `./code/demo.py`, **you must first download the models** by running:

```bash
cd model_cache
bash download_model.sh
```

---

## Demo

We provide a demonstration code to show the voxel-wise encoding model and BrainCLIP model using a simulated dataset.

### Running the Demo

After downloading the pretrained models, run the demo with:

```bash
cd code
python demo.py
```

### Expected Ouput of Demo

Note: this expected output is based on my environment. The finally output may vary depending on users' configuration.

```yaml
Current device is  cuda
----------Demo code for voxel-wise encoding model----------
Loading vision model config from /public/home/lishr2022/anaconda3/envs/tats/lib/python3.8/site-packages/cn_clip/clip/model_configs/ViT-H-14.json
Loading text model config from /public/home/lishr2022/anaconda3/envs/tats/lib/python3.8/site-packages/cn_clip/clip/model_configs/RoBERTa-wwm-ext-large-chinese.json
Model info {'embed_dim': 1024, 'image_resolution': 224, 'vision_layers': 32, 'vision_width': 1280, 'vision_head_width': 80, 'vision_patch_size': 14, 'vocab_size': 21128, 'text_attention_probs_dropout_prob': 0.1, 'text_hidden_act': 'gelu', 'text_hidden_dropout_prob': 0.1, 'text_hidden_size': 1024, 'text_initializer_range': 0.02, 'text_intermediate_size': 4096, 'text_max_position_embeddings': 512, 'text_num_attention_heads': 16, 'text_num_hidden_layers': 24, 'text_type_vocab_size': 2}
Shape of encoding model prediction is torch.Size([1, 10000])
Regulation of 8 layers is -2.0789623260498047
----------Demo code for BrainCLIP model----------
In BrainCLIP demo, the similarity between caption response and image response is 0.7341265678405762
```

---

## Citation
This repository was released with the following pre-print. If you use this repository in your research, please cite as:

Li, S., Jin, Z., Zhang, R. Y., Gu, S., & Li, Y. (2025). Predictive vision-language integration in the human visual cortex. bioRxiv, 2025-11. https://doi.org/10.1101/2025.11.03.686222

### **BibTeX**
```bibtex
@article{li2025predictive,
  title={Predictive vision-language integration in the human visual cortex},
  author={Li, Shurui and Jin, Zheyu and Zhang, Ru-Yuan and Gu, Shi and Li, Yuanning},
  journal={bioRxiv},
  pages={2025--11},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
---

## License
This repository is released under the **CC BY-NC-ND 4.0** license.  
You may use the code for research and educational purposes, but commercial use, distribution of modified versions, or sublicensing are not permitted.


For full terms, please refer to the [LICENSE](./LICENSE) file.






