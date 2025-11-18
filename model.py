import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel, AutoTokenizer, Wav2Vec2FeatureExtractor

class BanglaKeywordSpotter(nn.Module):
    def __init__(self):
        super(BanglaKeywordSpotter, self).__init__()
        
        # Load Audio Encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        # Load Audio Processor
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        # Load Text Encoder
        self.text_encoder = BertModel.from_pretrained("csebuetnlp/banglabert")
        
        # Load Text Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        
        # Freeze both encoders
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Define classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 768, 512),  # XLS-R dim + BERT dim -> hidden
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Output logits
        )
    
    def forward(self, audio_arrays, text_words):
        # Process Audio
        audio_inputs = self.audio_processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
        audio_inputs = {k: v.to(next(self.parameters()).device) for k, v in audio_inputs.items()}
        audio_outputs = self.audio_encoder(**audio_inputs)
        
        # Pooling Audio: (batch, time, 1024) -> (batch, 1024)
        # Changed from mean to max as requested
        audio_pool, _ = torch.max(audio_outputs.last_hidden_state, dim=1)
        
        # Process Text
        text_inputs = self.tokenizer(text_words, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}
        text_outputs = self.text_encoder(**text_inputs)
        
        # Pooling Text: Extract [CLS] token embedding (first token) -> (batch, 768)
        text_pool = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings
        
        # Concatenate audio and text features
        combined_features = torch.cat([audio_pool, text_pool], dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits.squeeze()