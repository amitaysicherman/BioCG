from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
import random
from dataclasses import dataclass

base_seed = 42
np.random.seed(base_seed)
random.seed(base_seed)
torch.manual_seed(base_seed)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Features:
    log_prob: float
    rank: float
    prob_over_random: float
    prob_to_next_largest: float
    entropy: float
    opt_count: float
    index: int


def features_list_to_dataframe(features_list):
    """
    Convert a list of Features objects to a pandas DataFrame.
    :param features_list: List of Features objects
    :return: DataFrame
    """
    data = [f.__dict__ for f in features_list]
    df = pd.DataFrame(data)
    return df


def get_sample_features(logits, labels):
    """
    Get the features of the samples from the single sample logits and labels.
    :param logits:  shape (seq_len,vocab_size)
    :param labels:  shape (seq_len) labels is -100 for padding
    """
    features_list = []
    for i in range(logits.shape[0]):
        if labels[i] == -100:
            continue
        logit = logits[i]
        log_probs = F.log_softmax(torch.tensor(logit), dim=0)
        probs = F.softmax(torch.tensor(logit), dim=0)
        non_zero_count = (~torch.isclose(log_probs, torch.tensor(-1e6))).sum()
        features = Features(
            log_prob=log_probs[labels[i]].item(),
            rank=torch.argsort(log_probs, descending=True).tolist().index(labels[i]),
            prob_over_random=log_probs[labels[i]].item() - (1 / non_zero_count),
            prob_to_next_largest=log_probs[labels[i]].item() - log_probs[
                torch.argsort(log_probs, descending=True)[1]].item(),
            entropy=-torch.sum(probs * log_probs).item(),
            opt_count=non_zero_count.item(),
            index=i
        )
        features_list.append([features.log_prob, features.rank, features.prob_over_random,
                              features.prob_to_next_largest, features.entropy, features.opt_count, features.index])
    return features_list


def get_batch_logits(model, batch):
    all_logits = []
    all_labels = []
    for batch in tqdm(batch, desc="Calculating logits", total=len(batch)):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        all_logits.append(output["logits"][:, :-1].detach().cpu().numpy())
        all_labels.append(batch["labels"][:, 1:].detach().cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels


def data_split_to_features(logits, labels):
    """
    Get the features of the samples from the logits and labels.
    :param logits:  shape (batch_size,seq_len,vocab_size)
    :param labels:  shape (batch_size,seq_len) labels is -100 for padding
    """
    features = []
    lens = []
    for i in range(logits.shape[0]):
        feature = get_sample_features(logits[i], labels[i])
        features.extend(feature)
        lens.append(len(feature))
    return np.array(features), np.array(lens)


def prepare_sequence_data(features, lengths, labels, max_seq_length):
    """
    Prepare sequence data by grouping features by their sequence and padding to max_seq_length
    """
    seq_features = []

    # Group features by their sequence
    start_idx = 0
    for length in lengths:
        if length == 0:
            continue
        # Extract features for this sequence
        seq_feats = features[start_idx:start_idx + length]
        # Pad to max_seq_length
        padded_feats = np.zeros((max_seq_length, seq_feats.shape[1]))
        padded_feats[:length] = seq_feats
        seq_features.append(padded_feats)
        start_idx += length

    return np.array(seq_features), labels[:len(seq_features)]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=2, num_layers=1, dropout=0.0, max_seq_length=256):

        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding - learnable
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sequence mask for padding (1 for tokens to attend to, 0 for padded positions)
        self.register_buffer('sequence_mask', torch.ones(max_seq_length))

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        # x shape: [batch_size, seq_len, feature_dim]

        # Create attention mask based on sequence lengths if provided
        if lengths is not None:
            mask = torch.zeros(x.size(0), x.size(1), device=x.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1.0
            mask = mask.bool()
        else:
            mask = None

        # Project input features to model dimension
        x = self.input_projection(x)

        # Add positional encodings
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Apply transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=~mask if mask is not None else None)

        # Global average pooling over non-padded sequence elements
        if mask is not None:
            # Apply mask for accurate mean calculation
            encoded = encoded * mask.unsqueeze(-1)
            pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = encoded.mean(dim=1)

        # Apply classifier
        output = self.classifier(pooled)

        return output.squeeze(-1)


def create_data_loader(features, labels, batch_size, shuffle=True):
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    dataset = TensorDataset(features_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Training and evaluation functions
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, auc


def get_metrics(y_true, y_pred):
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics['auc'] = roc_auc_score(y_true, y_pred)
    metrics['ap'] = average_precision_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary)
    metrics['recall'] = recall_score(y_true, y_pred_binary)
    metrics['f1'] = f1_score(y_true, y_pred_binary)
    metrics['sensitivity'] = recall_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    return metrics


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    metrics = get_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def run_training(X_valid, y_valid, X_test, y_test, d_model=128,
                 nhead=2,
                 num_layers=1,
                 dropout=0.0,
                 batch_size=16,
                 learning_rate=0.0005,
                 weight_decay=1E-05,
                 num_epochs=100,
                 patience=15,
                 ):
    # Create data loaders
    valid_loader = create_data_loader(X_valid, y_valid, batch_size, shuffle=True)
    test_loader = create_data_loader(X_test, y_test, batch_size, shuffle=False)

    # Initialize model
    input_dim = X_valid.shape[2]  # Feature dimension
    model = TransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=num_epochs)
    # Train model
    best_valid_auc = 0
    best_model_state = None
    no_improve = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Train
        valid_loss, valid_auc = train_epoch(model, valid_loader, criterion, optimizer)

        # Evaluate on test set (using as validation)
        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion)
    return test_metrics


def evaluate_model(valid_pos_dataset, valid_neg_dataset, test_pos_dataset, test_neg_dataset, model, batch_size=32,
                   d_model=128,
                   nhead=2,
                   num_layers=1,
                   dropout=0.0,
                   learning_rate=0.0005,
                   weight_decay=1E-05,
                   num_epochs=100,
                   patience=15,
                   ):
    valid_pos_loader = DataLoader(valid_pos_dataset, batch_size=batch_size, shuffle=True)
    valid_neg_loader = DataLoader(valid_neg_dataset, batch_size=batch_size, shuffle=True)
    test_pos_loader = DataLoader(test_pos_dataset, batch_size=batch_size, shuffle=False)
    test_neg_loader = DataLoader(test_neg_dataset, batch_size=batch_size, shuffle=False)
    features = {}

    for split in ['valid', 'test']:
        if split == 'valid':
            pos_loader = valid_pos_loader
            neg_loader = valid_neg_loader
        else:
            pos_loader = test_pos_loader
            neg_loader = test_neg_loader

        pos_logits, pos_labels = get_batch_logits(model, pos_loader)
        neg_logits, neg_labels = get_batch_logits(model, neg_loader)

        pos_features, pos_len = data_split_to_features(pos_logits, pos_labels)
        neg_features, neg_len = data_split_to_features(neg_logits, neg_labels)

        all_features = np.concatenate([pos_features, neg_features], axis=0)
        all_len = np.concatenate([pos_len, neg_len], axis=0)
        labels = np.concatenate([np.ones(len(pos_len)), np.zeros(len(neg_len))], axis=0)

        features[split] = (all_features, all_len, labels)
    valid_features, valid_lengths, valid_labels = features["valid"]
    test_features, test_lengths, test_labels = features["test"]

    # Create sequence level data with padding
    max_seq_length = max(np.max(valid_lengths), np.max(test_lengths))

    X_valid, y_valid = prepare_sequence_data(valid_features, valid_lengths, valid_labels, max_seq_length)
    X_test, y_test = prepare_sequence_data(test_features, test_lengths, test_labels, max_seq_length)
    results = run_training(X_valid, y_valid, X_test, y_test, d_model, nhead, num_layers, dropout, batch_size,
                           learning_rate, weight_decay, num_epochs, patience)

    # evaluate only for log_prob mean
    test_log_prob_mean = []
    for i in range(X_test.shape[0]):
        logs_prob = X_test[i][:, 0]
    logs_prob = logs_prob[logs_prob != 0]
    test_log_prob_mean.append(logs_prob.mean())

    log_prob_results = get_metrics(test_labels, test_log_prob_mean)
    log_prob_results = {f"log_prob_{k}": v for k, v in log_prob_results.items()}
    results.update(log_prob_results)
    return results
