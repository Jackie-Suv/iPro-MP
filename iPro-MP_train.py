import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import pandas as pd
import numpy as np

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

# Ensure reproducibility: make results consistent across runs
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Dataset class
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, kmer_size=6, max_len=128):
        self.sequences = sequences
        self.labels = labels  # Promoter labels
        self.tokenizer = tokenizer  # Tokenizer
        self.kmer_size = kmer_size  # K-mer size
        self.max_len = max_len  # Maximum sequence length

    def __len__(self):
        # Return the size of the dataset, i.e., the number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        # Split the sequence and perform tokenization
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Split the sequence into k-mers
        kmers = [sequence[i:i+self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]
        encoded = self.tokenizer(kmers, is_split_into_words=True, padding='max_length', max_length=self.max_len, return_tensors='pt', truncation=True)

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DNABERT classification model
class DNABERTPromoterClassifier(nn.Module):
    def __init__(self):
        super(DNABERTPromoterClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./DNABERT-6')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)  # First fully connected layer
        self.layer_norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer (optional)
        self.layer_norm2 = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 2)  # Binary classification
        self.activation = nn.GELU()  # Activation function

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # Get final hidden states from BERT
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token for binary classification

        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits

# Train the model
def train_model(model, train_loader, val_loader, fold_num, epochs, lr, i, fold, optimizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if optimizer == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    best_metrics = None  # Store best metrics
    best_fpr = None
    best_tpr = None
    best_auc = 0  # Initialize AUC as 0
    best_epoch = -1  # Record best epoch

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader)}")

        # Validate the model
        metrics, fpr, tpr, auc_score = validate_model(model, val_loader)

        # Compare and update the best metrics
        if auc_score > best_auc:
            best_metrics = metrics
            best_fpr = fpr
            best_tpr = tpr
            best_auc = auc_score
            best_epoch = epoch + 1
            model_save_path = f'./Result/model/Species{i}_fold{fold}.pth'
            torch.save(model.state_dict(), model_save_path)
        else:
            continue

    return best_metrics, best_fpr, best_tpr, best_auc

# Validate the model
def validate_model(model, val_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions, true_labels = [], []
    probs_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs  # Model output logits

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get the probability of the positive class

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)
            probs_list.extend(probs)

    fpr, tpr, thresholds = roc_curve(true_labels, probs_list)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(true_labels, probs_list)
    auprc = auc(recall, precision)

    acc = accuracy_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f'Sn: {sn:.4f}, Sp: {sp:.4f}, ACC: {acc:.4f}')
    print(f'AUC: {auc_score:.4f}, AUPRC: {auprc:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}')

    metrics = {
        'Sn': sn,
        'Sp': sp,
        'ACC': acc,
        'AUC': auc_score,
        'AUPRC': auprc,
        'MCC': mcc,
        'F1': f1
    }
    return metrics, fpr, tpr, auc_score

# Five-fold cross-validation
def cross_validate(df, tokenizer, output_file, i, species, kmer_size=6, batch_size=64, epochs=100, lr=1e-5, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold = 0

    sequences = df['text'].values
    labels = df['label'].values

    results = []  # Store results of each fold
    mean_fpr = np.linspace(0, 1, 100)  # Interpolated mean FPR
    tprs = []  # Store interpolated TPR
    aucs = []  # Store AUC values
    all_roc_data = []  # Store all ROC data

    plt.figure()

    for train_index, val_index in kf.split(sequences):
        fold += 1
        print(f"Training fold {fold}/{num_folds}")

        train_seqs, val_seqs = sequences[train_index], sequences[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # Create dataset and data loader
        train_dataset = DNADataset(train_seqs, train_labels, tokenizer, kmer_size=kmer_size)
        val_dataset = DNADataset(val_seqs, val_labels, tokenizer, kmer_size=kmer_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        model = DNABERTPromoterClassifier()

        # Train and validate the model
        metrics, fpr, tpr, auc_score = train_model(model, train_loader, val_loader, fold, epochs, lr, i, fold, optimizer='adam')
        metrics['Fold'] = fold
        results.append(metrics)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_score)

        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='Fold %d (AUC = %0.3f)' % (fold, auc_score))

        roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_data['Fold'] = fold
        all_roc_data.append(roc_data)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label='Mean ROC (AUC = %0.3f)' % mean_auc, lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='navy')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(species)
    plt.legend(loc="lower right")
    plt.savefig('./Results/Figure/%d_ROC.svg' % i, dpi=300)
    plt.close()

    all_roc_df = pd.concat(all_roc_data, ignore_index=True)
    all_roc_df.to_csv('./Results/ROC_Data/Species%d_roc_data.csv' % i, index=False)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    metrics_mean = df_results.mean(axis=0).to_dict()
    if 'Fold' in metrics_mean:
        del metrics_mean['Fold']

    return metrics_mean

name_list = ["Acinetobacter baumannii ATCC 17978", # 1
            "Bradyrhizobium japonicum USDA 110",   # 2
            "Burkholderia cenocepacia J2315", # 3
            "Campylobacter jejuni RM1221", # 4
            "Campylobacter jejuni subsp. jejuni 81116", # 5
            "Campylobacter jejuni subsp. jejuni 81-176", # 6
            "Campylobacter jejuni subsp. jejuni NCTC 11168", # 7
            "Corynebacterium diphtheriae NCTC 13129", # 8
            "Corynebacterium glutamicum ATCC 13032", # 9
            "Escherichia coli str K-12 substr. MG1655", # 10
            "Haloferax volcanii DS2", # 11
            "Helicobacter pylori strain 26695", # 12
            "Nostoc sp. PCC7120",  # 13
            "Paenibacillus riograndensis SBR5", # 14
            "Pseudomonas putida KT2440",  # 15
            "Shigella flexneri 5a str. M90T", # 16
            "Sinorhizobium meliloti 1021", # 17
            "Staphylococcus aureus subsp. aureus MW2", # 18
            "Staphylococcus epidermidis ATCC 12228", # 19
            "Synechococcus elongatus PCC 7942", # 20
            "Thermococcus kodakarensis KOD1", # 21
            "Xanthomonas campestris pv. campestrie B100",  # 22
            "Bacillus subtilis subsp. subtilis str. 168"   #23
            ]


# Main function
def main():
    all_datasets_results = []  # Store average metrics of all datasets

    for i in range(1, 24):
        print("***********************************************************************************************")
        print("******************************** Training species %d *****************************************" % i)
        input_file = './Benchmark Dataset/Train/%d_train.csv' % i
        df = pd.read_csv(input_file)

        output_file = './Result/Species%d_5fold.csv' % i
        tokenizer = BertTokenizer.from_pretrained('./DNABERT-6')

        species = name_list[i - 1]
        metrics_mean = cross_validate(df, tokenizer, output_file, i, species)

        metrics_mean['Datasets'] = species
        all_datasets_results.append(metrics_mean)

    df_all_results = pd.DataFrame(all_datasets_results)
    df_all_results.to_csv('./Results/AllSpecies-5fold.csv', index=False)

if __name__ == "__main__":
    main()
