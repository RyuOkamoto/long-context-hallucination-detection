# Towards Long Context Hallucination Detection

This repository contains the sources used in the following paper. Please consider citing if you use these sources.

```
@inproceedings{
liu2025towards,
title={Towards Long Context Hallucination Detection},
author={Siyi Liu and Kishaloy Halder and Zheng Qi and Wei Xiao and Nikolaos Pappas and Phu Mon Htut and Neha Anna John and Yassine Benajiba and Dan Roth},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025}
}
```

data/ contains a `sample.json` file with a dummy data point to train/test hallucination detection models.

src/ contains the code for training and evaluation. To train, simply run `bash train.sh`. To evaluate, run `eval.py` with appropriate arguments (see `train.sh`).
