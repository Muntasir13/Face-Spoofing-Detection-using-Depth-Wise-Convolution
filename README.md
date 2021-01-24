# Face Spoofing Detection using Depth Wise Convolution
Pytorch implementation of Depth Wise Convolution to perform the task of Face Spoofing Detection. The code implementation was done following the paper [A Face Spoofing Detection Method Based on Domain Adaptation and Lossless Size Adaptation] (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9056475). The Domain Adaptation Layer (partial) and Lossless Size Adaptation methodologies were avoided in this implementation.

## Getting Started
### Installing
- Install PyTorch and dependencies from https://pytorch.org.

### Training
- Download any dataset according to your choice.
- Go to config.py and make necessary changes according to the helping texts given.
- Run the following command:
```bash
python run.py
```

### Inference
- Go to the liveness_testing folder
- Copy and paste the desired trained model in this directory
- Go to liveness_demo.py and change the model path
- Run the following command:
```bash
python liveness_demo.py
```
