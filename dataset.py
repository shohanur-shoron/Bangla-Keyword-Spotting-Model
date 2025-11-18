import librosa
from torch.utils.data import Dataset

def load_data(file_path):
    """
    Load data from file with UTF-8 encoding.
    Parse lines as path|word|label

    Args:
        file_path: Path to the data file

    Returns:
        List of tuples (path, word, label)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('|')
                if len(parts) == 3:
                    path, word, label = parts
                    data.append((path, word, float(label)))
    return data

class BanglaDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list: List of tuples containing (path, word, label)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, word, label = self.data_list[idx]

        # Load audio with librosa - using 16000 to match model's expected sampling rate
        audio_array, _ = librosa.load(path, sr=16000)

        # Ensure word is a standard Python unicode string
        word_string = str(word)

        # Convert label to float
        float_label = float(label)

        return audio_array, word_string, float_label