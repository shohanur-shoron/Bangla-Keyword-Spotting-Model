import torch
import numpy as np

def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of tuples (audio_array, word_string, float_label)
    
    Returns:
        audios: List of numpy arrays
        words: List of strings
        labels: torch.tensor of floats
    """
    audios = []
    words = []
    labels = []
    
    for audio_array, word_string, float_label in batch:
        audios.append(audio_array)
        words.append(word_string)
        labels.append(float_label)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.float)
    
    return audios, words, labels