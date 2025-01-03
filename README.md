# Instructions

This repo provides data and scripts to reproduce ressults in the following paper:

> Deriba, A. Z., & Yang, D. Y. (2025).
> Performance-Based Risk Assessment for Large-Scale Transportation Networks Using the Transitional Markov Chain Monte Carlo Method.
> _ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering._
> 11(1), 04024090.
> [DOI: https://doi.org/10.1061/AJRUA6.RUENG-1402](https://doi.org/10.1061/AJRUA6.RUENG-1402)


It also serves as an example and tutorial of using TMCMC for network risk assessment.

## Network generation

Run `prepare_OR_network.py` to generate graphs.

## Case I examples

For Case I examples, run `case_I.py` with different random seeds and asset numbers.

## Case II examples

For Case II examples, run `case_II.py` with different random seeds, asset numbers, and useful assets.

## OR network risk

The script `OR_risk.py` reproduces network risk of Oregon highway links using method proposed in the paper.
The script `OR_MCS_risk.py` reproduces network risk of Oregon highway links using crude MCS. 
