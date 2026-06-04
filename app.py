import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from pydub import AudioSegment
import gradio as gr

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_english_deep_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def convert_to_wav(in_path: str, out_path: str) -> str:
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

def remove_noise(y: np.ndarray, sr: int) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rmse])

def predict(audio_path: str) -> str:
    if audio_path is None:
        return "Please upload an audio file."
    try:
        wav_path = "converted.wav"
        convert_to_wav(audio_path, wav_path)

        y, sr = librosa.load(wav_path, sr=16000)
        y = remove_noise(y, sr)

        features = extract_features(y, sr)
        scaled_features = scaler.transform([features]).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], scaled_features)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = label_encoder.inverse_transform([np.argmax(output_data)])[0]

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return f"Prediction: {predicted_label}"

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload English Audio File"),
    outputs=gr.Text(label="Result"),
    title="TrueTone — English Audio Forgery Detection",
    description="Upload an English audio file to detect whether it is Original, AI Generated, or Combined. Built using a TFLite deep learning model trained on English audio samples.",
    examples=[],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
