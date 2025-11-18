import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import BanglaKeywordSpotter
from dataset import BanglaDataset
from collate import collate_fn

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

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data = load_data('bangla_data.txt')

    # Create dataset
    train_dataset = BanglaDataset(train_data)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # You can adjust batch size as needed
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = BanglaKeywordSpotter()
    model.to(device)

    # Optimizer: Only train the classifier parameters
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 10  # You can adjust the number of epochs as needed
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (audios, words, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass - pass entire batch to model
            outputs = model(audios, words)
            labels = labels.to(device)

            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'bangla_kws_model.pth')
    print("Model saved as 'bangla_kws_model.pth'")

if __name__ == "__main__":
    main()