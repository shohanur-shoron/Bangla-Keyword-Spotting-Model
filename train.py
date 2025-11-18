import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model import BanglaKeywordSpotter
from dataset import BanglaDataset, load_data
from collate import collate_fn
import argparse

def calculate_metrics(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_class)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_class, average='binary')
    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train the Bangla Keyword Spotting Model.")
    parser.add_argument('--train_data', type=str, default='bangla_data.txt', help='Path to the training data file.')
    parser.add_argument('--val_data', type=str, default=None, help='Path to the validation data file.')
    parser.add_argument('--model_path', type=str, default='bangla_kws_model.pth', help='Path to save the trained model.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data = load_data(args.train_data)
    train_dataset = BanglaDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    if args.val_data:
        val_data = load_data(args.val_data)
        val_dataset = BanglaDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    # Initialize model
    model = BanglaKeywordSpotter()
    model.to(device)

    # Optimizer: Only train the classifier parameters
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for batch_idx, (audios, words, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(audios, words)
            labels = labels.to(device)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Store predictions and labels for metrics
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(all_train_labels, all_train_preds)
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        print(f'Train Metrics: Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')

        if args.val_data:
            model.eval()
            all_val_labels = []
            all_val_preds = []
            with torch.no_grad():
                for audios, words, labels in val_loader:
                    outputs = model(audios, words)
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    all_val_preds.extend(preds)
                    all_val_labels.extend(labels.numpy())

            val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_val_labels, all_val_preds)
            print(f'Validation Metrics: Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved as '{args.model_path}'")

if __name__ == "__main__":
    main()