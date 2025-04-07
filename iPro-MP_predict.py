import os
import argparse
import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn

from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import warnings
import random
import logging

warnings.filterwarnings('ignore')


# List of species names
name_list = ["Acinetobacter baumannii ATCC 17978", 
             "Bradyrhizobium japonicum USDA 110", 
             "Burkholderia cenocepacia J2315", 
             "Campylobacter jejuni RM1221", 
             "Campylobacter jejuni subsp. jejuni 81116", 
             "Campylobacter jejuni subsp. jejuni 81-176", 
             "Campylobacter jejuni subsp. jejuni NCTC 11168", 
             "Corynebacterium diphtheriae NCTC 13129", 
             "Corynebacterium glutamicum ATCC 13032", 
             "Escherichia coli str K-12 substr. MG1655", 
             "Haloferax volcanii DS2", 
             "Helicobacter pylori strain 26695", 
             "Nostoc sp. PCC7120", 
             "Paenibacillus riograndensis SBR5", 
             "Pseudomonas putida KT2440", 
             "Shigella flexneri 5a str. M90T", 
             "Sinorhizobium meliloti 1021", 
             "Staphylococcus aureus subsp. aureus MW2", 
             "Staphylococcus epidermidis ATCC 12228", 
             "Synechococcus elongatus PCC 7942", 
             "Thermococcus kodakarensis KOD1", 
             "Xanthomonas campestris pv. campestrie B100", 
             "Bacillus subtilis subsp. subtilis str. 168"]

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Predict DNA sequence classification using DNABERT.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Input FASTA file containing DNA sequences.")
    parser.add_argument('-s', '--species_id', type=int, required=True, choices=range(1, 24), help="Species ID (1-23) corresponding to the species list.")
    parser.add_argument('-o', '--output_file', type=str, default=None, help="Output file name for saving the prediction results.")
    return parser.parse_args()

# Create output directory
def create_output_dir():
    output_dir = './Predict_Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# -----------------------------
# 1. Dataset class: DNADataset
# -----------------------------
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, kmer_size=6, max_len=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.kmer_size = kmer_size
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Split sequence into k-mers
        kmers = [sequence[i:i+self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]
        
        # Encode using tokenizer
        encoded = self.tokenizer(
            kmers, 
            is_split_into_words=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt', 
            truncation=True
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),          # Shape: (max_len)
            'attention_mask': encoded['attention_mask'].squeeze(),# Shape: (max_len)
            'label': torch.tensor(label, dtype=torch.long)
        }

# ---------------------------------------------
# 2. Model class: DNABERTPromoterClassifier
# ---------------------------------------------
class DNABERTPromoterClassifier(nn.Module):
    """
    A classification model based on pre-trained DNABERT, including additional fully connected layers, layer normalization, Dropout, and GELU activation function.
    """
    def __init__(self):
        super(DNABERTPromoterClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./DNABERT-6')  # Load the pre-trained DNABERT model
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)  # First fully connected layer
        self.layer_norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer (optional)
        self.layer_norm2 = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 2)  # Binary classification output layer
        self.activation = nn.GELU()  # Activation function
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Get the output of the [CLS] token
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

# ---------------------------------------------
# 3. Prediction function: predict_with_model
# ---------------------------------------------
def predict_with_model(species_id, input_file, output_dir, output_file=None, model_dir='./models/', kmer_size=6, max_len=128, batch_size=64):
    # Select species name
    species_name = name_list[species_id - 1]

    # Load the pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained('./DNABERT-6')

    # Load test sequences
    with open(input_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines() if not line.startswith(">")]

    # Create test dataset and dataloader
    test_dataset = DNADataset(sequences=sequences, labels=[0]*len(sequences), tokenizer=tokenizer, kmer_size=kmer_size, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically use GPU if available
    models = []
    for fold in range(1, 6):
        model_path = os.path.join(model_dir, f"{species_id}_fold_{fold}.pth")
        model = DNABERTPromoterClassifier()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Prediction
    aggregated_probs = np.zeros(len(test_loader.dataset))
    with torch.no_grad():
        for model_idx, model in enumerate(models):
            probs = []
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of positive class
                probs.extend(probabilities.cpu().numpy())

            aggregated_probs += np.array(probs)

    # Average probabilities
    aggregated_probs /= len(models)
    aggregated_preds = (aggregated_probs >= 0.5).astype(int)

    # Determine output file name
    if output_file is None:
        output_file = os.path.join(output_dir, f"{species_id}_predictions.csv")
    else:
        output_file = os.path.join(output_dir, output_file)

    # Save prediction results
    results = pd.DataFrame({'Sequence': sequences, 'Prediction': aggregated_preds, 'Probability': aggregated_probs})
    results.to_csv(output_file, index=False)

# Main function
def main():
    args = parse_args()

    # Create output folder
    output_dir = create_output_dir()

    # Execute prediction
    predict_with_model(args.species_id, args.input_file, output_dir, args.output_file)

if __name__ == "__main__":
    main()
