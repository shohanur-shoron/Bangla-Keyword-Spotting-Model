# Bangla Keyword Spotting Model

## Overview

This project implements a **Bangla (Bengali) Keyword Spotting Model** that can detect specific Bengali words in audio recordings. The model combines audio processing with text understanding to identify when a particular Bengali word appears in spoken audio.

## What the Model Does

This model is designed to:
- Take an audio recording and a Bengali keyword as input
- Determine if the specified keyword is present in the audio
- Return a confidence score indicating the likelihood of the keyword being present

Think of it as a "Bengali word detector" for audio - similar to how you might search for text in a document, but for spoken words in audio recordings.

## How It Works

The model uses a **multimodal approach**, meaning it processes both audio and text simultaneously:

1. **Audio Processing**: Uses Wav2Vec2 (pre-trained on speech) to analyze the audio content
2. **Text Processing**: Uses BanglaBERT (a model trained specifically for Bengali text) to understand the meaning of the keyword
3. **Feature Combination**: Combines both audio and text representations
4. **Classification**: Determines if the keyword matches what's in the audio

The model is trained to recognize the relationship between how Bengali words sound and their textual representation.

## Requirements

### Python Dependencies
```bash
torch
torchaudio
transformers
librosa
numpy
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

### Pre-trained Models
During execution, the model will automatically download:
- `facebook/wav2vec2-large-xlsr-53`: For audio processing
- `csebuetnlp/banglabert`: For Bengali text processing

## Project Structure

```
KWDB/
├── model.py            # Defines the BanglaKeywordSpotter neural network
├── dataset.py          # Handles loading audio data
├── collate.py          # Batches data for training
├── train.py            # Training script
├── predict.py          # Prediction script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setting Up Your Data

The model expects training data in a text file named `bangla_data.txt` with the following format:

```
/path/to/audio1.wav|word_to_spot|label
/path/to/audio2.wav|another_word|label
```

Where:
- First column: Path to the audio file
- Second column: The Bengali word/phrase to spot
- Third column: Label (1 for present, 0 for not present)

Example:
```
audio/speech1.wav|হ্যালো|1
audio/speech2.wav|হ্যালো|0
audio/speech3.wav|ধন্যবাদ|1
```

## Training the Model

1. **Prepare Your Data**: Create a `bangla_data.txt` file with your training data in the format described above.

2. **Run Training**:
```bash
python train.py
```

The training process will:
- Load your audio-text pairs from `bangla_data.txt`
- Train only the classification layer (the pre-trained encoders remain frozen)
- Save the trained model as `bangla_kws_model.pth`

### Training Details
- Only the classifier parameters are updated during training
- The pre-trained audio and text encoders are frozen
- Uses AdamW optimizer with learning rate 1e-4
- Uses Binary Cross-Entropy loss for the binary classification task

## Using the Trained Model for Prediction

After training, you can use the model to detect keywords in new audio files:

```bash
python predict.py
```

The script will prompt you to enter:
1. Path to the audio file you want to analyze
2. The Bengali keyword you're looking for

### Example Usage:
```bash
Enter the path to the audio file: test_audio.wav
Enter the keyword to detect: হ্যালো
```

The output will show:
- Logit score (raw output from the model)
- Probability (confidence between 0 and 1)
- Prediction (whether the keyword was detected)

## Programmatic Usage

You can also use the model programmatically in your Python code:

```python
import torch
import librosa
from model import BanglaKeywordSpotter

# Load the trained model
model = BanglaKeywordSpotter()
model.load_state_dict(torch.load('bangla_kws_model.pth'))
model.eval()

# Load an audio file 
audio_array, _ = librosa.load('audio.wav', sr=16000)
audio_batch = [audio_array]
keyword_batch = ['আপনি']  # The Bengali keyword to detect

# Make prediction
with torch.no_grad():
    logit = model(audio_batch, keyword_batch)
    probability = torch.sigmoid(logit).item()
    prediction = 1 if probability > 0.5 else 0  # Using 0.5 threshold

print(f"Probability: {probability}")
print(f"Prediction: {'Present' if prediction == 1 else 'Not Present'}")
```

## Model Architecture

The `BanglaKeywordSpotter` model consists of:

1. **Audio Encoder**: `facebook/wav2vec2-large-xlsr-53` (frozen)
   - Converts audio to 1024-dimensional feature vectors
   - Pre-trained on multiple languages including some South Asian languages

2. **Text Encoder**: `csebuetnlp/banglabert` (frozen)
   - Converts text to 768-dimensional feature vectors
   - Specifically trained for Bengali language understanding

3. **Classifier**: A simple neural network with:
   - Input: 1024 (audio) + 768 (text) = 1792 dimensions
   - Hidden layer: 512 neurons with ReLU activation
   - Dropout: 30% for regularization
   - Output: 1 neuron (for binary classification)

## Customization Options

### Changing the Threshold
In `predict.py`, you can adjust the classification threshold:
```python
logit, probability, prediction = predict_keyword(audio_file, keyword, model, threshold=0.3)  # Lower threshold = more sensitive
```

### Adjusting Training Parameters
In `train.py`, you can modify:
- Batch size in the DataLoader
- Number of epochs
- Learning rate in the optimizer
- Loss function if needed

## Expected Performance

This model is optimized for:
- Detecting specific Bengali keywords in audio recordings
- Working with clear audio (works best with good quality recordings)
- Processing audio with a sampling rate of 16kHz (though it can handle other rates)

## Troubleshooting

### Common Issues:

1. **Memory Errors**: Try reducing the batch size in `train.py`
2. **Audio Loading Issues**: Ensure audio files are in common formats (WAV, MP3, etc.)
3. **Model Loading Errors**: Check that the model file `bangla_kws_model.pth` exists
4. **Unicode Issues with Bengali Text**: The code handles UTF-8 encoding properly

### Model File Size
The saved model (`bangla_kws_model.pth`) only contains the classifier weights, not the pre-trained encoders, so it should be relatively small.

## Use Cases

This model is suitable for applications such as:
- Voice command detection in Bengali
- Monitoring Bengali speech for specific keywords
- Accessibility tools for Bengali speakers
- Content filtering for Bengali audio
- Speech analysis and research

## Limitations

- Requires training data with labeled audio-text pairs
- Performance depends on audio quality
- May have difficulty with heavily accented speech or background noise
- The pre-trained encoders are fixed, so model adapts to new keywords but not new audio characteristics