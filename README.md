# Signature Forgery Detection using HSSAE (CNN + Siamese Network)

A deep learning project that classifies signatures as genuine or forged, to replace slow and error-prone manual verification.

## Overview
- **Model:** HSSAE, built with a CNN and a Siamese Network
- **Task:** verify whether a signature is genuine or forged
- **Result:** 95% verification accuracy
- **Tech stack:** Python, OpenCV, TensorFlow, CNN, Siamese Network

## Workflow
1. **Preprocessing:** grayscale conversion and noise removal using OpenCV
2. **Feature extraction:** features extracted from the preprocessed signature images
3. **Training:** CNN and Siamese Network trained for 50 epochs using TensorFlow
4. **Evaluation:** verification accuracy of 95%

## Project Structure
```
signature_gui.py   - data processing script
test/              - test files
```

## Requirements
- Python
- TensorFlow
- OpenCV (opencv-python)

## Author
Bharath S | [LinkedIn](https://linkedin.com/in/bharaths003) | [GitHub](https://github.com/bharathsenthilkumar0103)
