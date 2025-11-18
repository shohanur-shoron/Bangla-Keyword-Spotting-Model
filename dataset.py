import librosa
from torch.utils.data import Dataset

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