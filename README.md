# Recurrent-Generative Welding Model with Integrated Waveform Data

## Overview

This repository contains the implementation of a Generative Adversarial Network (GAN) for modeling imaged welding process dynamic behaviors. The model integrates waveform data to establish a foundation for monitoring weld penetration using deep learning techniques.

## Requirements

- Python 3.7
- Required packages can be installed using:
  ```
  pip install -r requirements.txt
  ```

## Data Format

The model works with welding images that have been processed to the following specifications:
- Dimensions: 64x64x1
- Format: BMP

## Usage Instructions

### Data Structure
Your data directory should have the following structure:
```
data_directory/
├── real/           # Contains topside welding images
├── noise/          # Contains bottom welding images
└── current_new.csv # Contains waveform current information
```

### Running the Model
1. Modify the hyperparameters in the `run.sh` file as needed
2. Update the `--train_data_dir` parameter in `run.sh` to point to your data directory
3. Execute the following command to run the model:
   ```
   bash run.sh
   ```

## Citation

If you use this code or our results in your research, please cite as appropriate:

```
@article{mucllari2024modeling,
  title={Modeling imaged welding process dynamic behaviors using generative adversarial network (GAN) for a new foundation to monitor weld penetration using deep learning},
  author={Mucllari, Edison and Cao, Yue and Ye, Qiang and Zhang, YuMing},
  journal={Journal of Manufacturing Processes},
  volume={124},
  pages={187--195},
  year={2024},
  publisher={Elsevier}
}
```
