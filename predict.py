import torch
import librosa
from model import BanglaKeywordSpotter

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
    # Load the trained model
    model = load_model('bangla_kws_model.pth')
    
    # Example usage
    audio_file = input("Enter the path to the audio file: ")
    keyword = input("Enter the keyword to detect: ")
    
    logit, probability, prediction = predict_keyword(audio_file, keyword, model)
    
    print(f"\nResults for audio: {audio_file}")
    print(f"Keyword: {keyword}")
    print(f"Logit: {logit:.4f}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction: {'Present' if prediction == 1 else 'Not Present'}")
    
    # You can also use it programmatically like this:
    # audio_path = "path/to/your/audio.wav"
    # keyword = "আপনি"
    # logit, probability, prediction = predict_keyword(audio_path, keyword, model)
    # print(f"Keyword '{keyword}' is {'present' if prediction == 1 else 'not present'} in the audio.")

if __name__ == "__main__":
    main()