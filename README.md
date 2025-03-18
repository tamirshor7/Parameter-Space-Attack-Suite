# Adversarial Attacks for Weight-Space Classifiers

This repository contains the official implementation of the paper:

**["Adversarial Attacks for Weight-Space Classifiers"](https://arxiv.org/abs/2502.20314)**  
Accepted as a **spotlight** at the [ICLR 2025 Workshop on Weight-Space Learning](https://weight-space-learning.github.io/).

## Installation

To set up the environment, install the dependencies using Conda:

```bash
conda env create -f environment.yml
conda activate pss
```

## Usage & Result Reproduction

In the following sections we provide instructions for full recreation of results from the paper for each of our three considered baselines - MNIST, Fashion-MNIST and ModelNet10. \\
Our experimental pipeline is compsoed of 3 stages -in section [#Functaset Creation] we provide information on creating our datasets of SIREN modulation vectors according [to Dupont et al](https://arxiv.org/abs/2201.12204).
These datasets are used for the 'clean' training of the weight-space classifiers (without incorporation of adversarial attacks). Insturctions for performing 'clean' classifier training are in section [#'Clean' Classifier Training].
Finally, we provide usage examples and instructions for paper-results recreation for each attack in our suite in section [#Adversarial Attacks]. 


### Functaset Creation
1. **Train a Functa as a meta-learner:**
   - MNIST:
     ```bash
     python trainer.py --dataset mnist
     ```
   - Fashion-MNIST:
     ```bash
     cmd1_fmnist
     ```
   - ModelNet10:
     ```bash
     python trainer.py --dataset modelnet --data-path /home/tamir.shor/adv_shapenet/ModelNet10 --batch-size 32 --num-epochs 20 --mod-dim 2048
     ```

2. **Create the Functaset:**
   - MNIST:
     ```bash
     cmd2_mnist
     ```
   - Fashion-MNIST:
     ```bash
     cmd2_fmnist
     ```
   - ModelNet10:
     ```bash
     cmd2_modelnet10
     ```

### 'Clean' Classifier Training
### Adversarial Attacks
1. **Train a Functa as a meta-learner:**
   ```bash
   cmd1
   ```
2. **Create the Functaset:**
   ```bash
   cmd2
   ```
3. **Train a weight-space classifier:**
   ```bash
   cmd3
   ```
4. **Run adversarial attacks:**
   - Attack 1:
     ```bash
     cmd4
     ```
   - Attack 2:
     ```bash
     cmd5
     ```
   - Attack 3:
     ```bash
     cmd6
     ```
   - Attack 4:
     ```bash
     cmd7
     ```

### Special Case: L-BFGS Attack
For the fifth attack type, the entire pipeline is trained with L-BFGS:
1. **Create the Functaset:**
   ```bash
   cmd8
   ```
2. **Train the classifier:**
   ```bash
   cmd9
   ```
3. **Run the attack:**
   ```bash
   cmd10
   ```

## Acknowledgements
This code is adapted from [Original Repository](#) (add link here if available). We thank the original authors for their contributions.

## Citation
If you find this work useful, please consider citing our paper:

```bibtex
@article{your_paper,
  author    = {Your Name and Co-authors},
  title     = {Adversarial Attacks for Weight-Space Classifiers},
  journal   = {arXiv preprint arXiv:2502.20314},
  year      = {2025}
}
