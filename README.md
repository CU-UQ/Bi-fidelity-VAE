# Bi-fidelity Variational Auto-encoder

This repo contains code for our paper 
> N. Cheng, O. A. Malik, S. Becker, A. Doostan.
> *Bi-fidelity Variational Auto-encoder for Uncertainty Quantification*.
> **arXiv preprint arXiv:2305.16530**,
> 2023.

The paper is available at [arXiv](https://arxiv.org/abs/2305.16530) and [DOI](https://doi.org/10.1016/j.cma.2024.116793).

## Referencing this code

If you use this code in any of your own work, please reference our paper:
```
@article{cheng2024bi,
  title={Bi-fidelity variational auto-encoder for uncertainty quantification},
  author={Cheng, Nuojin and Malik, Osman Asif and De, Subhayan and Becker, Stephen and Doostan, Alireza},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={421},
  pages={116793},
  year={2024},
  publisher={Elsevier}
}
```

## Description of code 
### Main file 

- **models.py**: contain a bi-fidelity data class BFDataset, a fully-connected variational auto-encoder class, and a fully-connected bi-fidelity variational auto-encoder; 

### Helper file

- **MMD_func.py**: compute the maximum mean discrepancy (MMD) with a mixture of rational quadratic kernels;

### Experiment files

- **BF-VAE-beam.ipynb**: BF-VAE implementation on the composite beam model with MMD results;

- **BF-VAE-cav.ipynb**: BF-VAE implementation on the cavity flow model with MMD results;

- **BF-VAE-burgers.ipynb**: BF-VAE implementation on the 1D visous Burgers model with MMD results.

## Data

The bi-fidelity data is available in this [link](https://zenodo.org/record/10263107). 

## Author contact information

Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on the paper. I can be reached at `nuojin (dot) cheng (at) colorado (dot) edu`. 

