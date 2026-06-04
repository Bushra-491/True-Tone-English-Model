TrueTone — English Audio Forgery Detection
🎙️ A deep learning system to detect whether an Urdu audio file is Original, AI Generated, or Combined.

🔴 Live Demo https://huggingface.co/spaces/BushraSaleem491/TrueTone-English

Overview
This project was built as a Final Year Project for BS Information Technology at the University of Gujrat. It is a complete end-to-end pipeline for audio forgery detection in the  English language — one of systems for English audio.

Pipeline
1. Dataset Curation

Manually curated 1,530+ Urdu audio files
Divided into Training, Validation, and Testing  files
Three categories: Original, AI Generated, Combined
2. Preprocessing

Converted all audio to WAV format at 16kHz mono
Volume normalization using librosa
3. Data Augmentation

Generated 10,710+ augmented files using 7 techniques:
Pitch shift
Speed up (1.2x)
Slow down (0.8x)
White noise addition
Random volume change
Reverb effect
Band-pass filtering
4. Feature Extraction

MFCC (13 features)
Chroma (12 features)
Spectral Contrast (7 features)
Zero Crossing Rate (1 feature)
RMSE (1 feature)
Total: 34 features per audio file
5. Model Training

Architecture: Dense Neural Network (256 → 128 → output)
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epochs: 100
6. Deployment

Converted to TFLite for lightweight inference
Deployed to Hugging Face Spaces with Gradio interface
Tech Stack
Python, TensorFlow, librosa, scikit-learn, pydub, scipy, Gradio, Hugging Face Spaces

Results
Model trained and evaluated on held-out test set of 170 files across 3 categories.

Try It Live
https://huggingface.co/spaces/BushraSaleem491/TrueTone-English
