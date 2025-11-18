import torch
import librosa
from model import BanglaKeywordSpotter
import argparse
import os

def load_model(model_path):
    """
    Load the trained model from the specified path.
    
    Args:
        model_path: Path to the saved model state dict
        
    Returns:
        Loaded model in evaluation mode
    """
    model = BanglaKeywordSpotter()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_keyword(audio_path, keyword, model, threshold=0.5):
    """
    Predict if the given keyword is present in the audio.

    Args:
        audio_path: Path to the audio file
        keyword: The keyword to spot in the audio
        model: Loaded trained model
        threshold: Threshold for classification (default 0.5)

    Returns:
        Tuple of (logit, probability, prediction)
    """
    # Load audio - using 16000 Hz to match the training data
    audio_array, _ = librosa.load(audio_path, sr=16000)

    # Prepare audio as a batch
    audio_batch = [audio_array]
    keyword_batch = [keyword]

    # Move to appropriate device
    device = next(model.parameters()).device

    with torch.no_grad():
        # Get prediction
        logit = model(audio_batch, keyword_batch)

        # Convert logit to probability using sigmoid
        probability = torch.sigmoid(logit).item()

        # Make prediction based on threshold
        prediction = 1 if probability > threshold else 0

    return logit.item(), probability, prediction

def main():
    parser = argparse.ArgumentParser(description="Predict a keyword in a Bangla audio file.")
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file.')
    parser.add_argument('--keyword', type=str, required=True, help='The Bengali keyword to detect.')
    parser.add_argument('--model_path', type=str, default='bangla_kws_model.pth', help='Path to the trained model.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification.')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found at '{args.audio_file}'")
        return

    # Load the trained model
    model = load_model(args.model_path)
    
    logit, probability, prediction = predict_keyword(args.audio_file, args.keyword, model, args.threshold)
    
    print(f"\nResults for audio: {args.audio_file}")
    print(f"Keyword: {args.keyword}")
    print(f"Logit: {logit:.4f}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction: {'Present' if prediction == 1 else 'Not Present'}")

if __name__ == "__main__":
    main()