# Locally-Rank-One-Based Joint Unmixing and Demosaicing for Snapshot Spectral Images

This repository contains a MATLAB implementation for the methods described in the following article:

**K. Abbas, M. Puigt, G. Delmaire, and G. Roussel (2024).**  
*"Locally-Rank-One-Based Joint Unmixing and Demosaicing Methods for Snapshot Spectral Images. Part I: A Matrix-Completion Framework."*  
*IEEE Transactions on Computational Imaging, 10, 848-862.*  
DOI: [10.1109/TCI.2024.3402322](https://doi.org/10.1109/TCI.2024.3402322)

## Overview
This project implements a matrix-completion framework for joint unmixing and demosaicing of snapshot spectral images, leveraging locally rank-one approximations and sparsity techniques as outlined in the referenced article. The code is provided under the MIT License (see below).

## Requirements
- MATLAB (version R2020 or later recommended)
- To reproduce all the figures of the paper, you need to download the other methods functions: `PPID`, `GRMR`, `ItSD`, etc... These can be found at [https://github.com/gtsagkatakis/Snapshot_Spectral_Image_demosaicing](https://github.com/gtsagkatakis/Snapshot_Spectral_Image_demosaicing).

## Running
To test all the methods, run `main.m`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgments
If you use this code in your research please cite the article :
@ARTICLE{10535201,
author={Abbas, Kinan and Puigt, Matthieu and Delmaire, Gilles and Roussel, Gilles},
journal={IEEE Transactions on Computational Imaging},
title={Locally-Rank-One-Based Joint Unmixing and Demosaicing Methods for Snapshot Spectral Images. Part I: A Matrix-Completion Framework},
year={2024},
volume={10},
pages={848-862},
doi={10.1109/TCI.2024.3402322}
}


## Contact
For questions or issues, please contact Kinan Abbas at kinan.abbas@ens-lyon.fr or Kinan.3bbas@gmail.com