# Signature Forgery Detection using HSSAE (CNN + Siamese Network)

A deep learning project that classifies signatures as genuine or forged, to replace slow and error-prone manual verification.

## Overview
- Model: HSSAE, built with a CNN and a Siamese Network
- Task: verify whether a signature is genuine or forged
- Result: 95% verification accuracy
- Tech stack: Python, OpenCV, CNN, Siamese Network, [TensorFlow / Keras / PyTorch: whichever you used]

## Dataset
- Name: [dataset name, e.g. CEDAR / your own collected samples]
- Size: [number of signers and number of genuine/forged samples]

## Workflow
1. **Preprocessing:** [e.g. grayscale conversion, resizing, noise removal using OpenCV]
2. **Feature extraction:** [what you did]
3. **Training:** [how the CNN + Siamese network was trained, number of epochs]
4. **Evaluation:** accuracy of 95% on [test set description]

## Project Structure
```
signature_gui.py   - [what it does, e.g. GUI to upload and verify a signature]
test/              - [what is in this folder]
```

## How to Run
```
pip install -r requirements.txt
python signature_gui.py
```

## Results
- Verification accuracy: 95%

## Author
Bharath S | [LinkedIn](https://linkedin.com/in/bharaths003) | [GitHub](https://github.com/bharathsenthilkumar0103)
