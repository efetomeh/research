# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
xds = pd.read_excel('/kaggle/input/teefive-dataset/yep.xlsx')
#yds = pd.read_excel('/kaggle/input/abcde/ugh-1.xlsx')
#concat_df = pd.concat([xds,yds]).reset_index(drop = True)
df = xds.copy()
#for idx, columnx in enumerate(df): print(idx, columnx)
import ast
df['_relationship'] = df['relationships'].apply(lambda x: ast.literal_eval(x))
df.describe()
!pip install -q spacy nltk SentencePiece spacy_transformers numpy transformers
#!python -m spacy download en_core_web_trf
import spacy, nltk, string, os, gc, torch, spacy_transformers, transformers
#nlp = spacy.load('en_core_web_trf')
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, T5Config, get_linear_schedule_with_warmup
# Download the English models for Spacy


#Loading the English models for Spacy

# Split the dataset into training, validation, and testing sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_size = int(train_ratio * len(df))
val_size = int(val_ratio * len(df))
test_size = len(df) - train_size - val_size

train_df = df[:train_size].reset_index(drop = True)
val_df = df[train_size:train_size + val_size].reset_index(drop = True)
test_df = df[train_size + val_size:].reset_index(drop = True)
"""x_relationships = df['_relationship'][400]
xinput_text = ' '.join([
            f"{relationships['token']}_{relationships['child']}_{relationships['dependency']}_{relationships['semantic_relationship']}"  for relationships in x_relationships])
print(len(tokenizer.encode(xinput_text, add_special_tokens = False)))"""
# Define a custom dataset class with semantic labels
class RelationshipRolesDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=5000):
        self.data = df['_relationship']
        self.target = df[df.columns[4]]
        self.tokenizer = tokenizer
        self.max_length = max_length  # Set the dynamic max_length here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _relationships = self.data[index]
        input_text = ' '.join([
            f"{relationships['token']}_{relationships['child']}_{relationships['dependency']}_{relationships['semantic_relationship']}"  for relationships in _relationships])  # Concatenate relationships and text
        target_text = self.target[index]

        # Calculate the actual length of the input text
        input_text_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))

        # Set the dynamic max_length based on the input text length
        dynamic_max_length = min(input_text_length + 2, self.max_length)  # Add 2 for [CLS] and [SEP] tokens

        input_encoding = self.tokenizer.encode_plus(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=dynamic_max_length,  # Use dynamic max_length here
            return_tensors='pt'
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
        }


def custom_collate(batch):
    """A custom collate function that pads tensors to the maximum sequence length in the batch.

    Args:
        batch: A list of dictionaries, where each dictionary contains tensors.

    Returns:
        A dictionary of tensors, with the same keys as the input dictionaries, padded to the maximum sequence length in the batch.
    """

    # Find the maximum sequence length in the current batch for each key
    max_lengths = {key: max(sample[key].shape[-1] for sample in batch) for key in batch[0].keys()}

    # Initialize the output dictionary
    output = {}

    for key in batch[0].keys():
        # Pad each tensor in the batch to the maximum sequence length in the batch
        padded_tensors = [torch.nn.functional.pad(sample[key], pad=(0, max_lengths[key] - sample[key].shape[-1])) for sample in batch]

        # Stack the padded tensors along a new dimension to create a batch tensor
        stacked_tensor = torch.stack(padded_tensors)

        # Store the stacked tensor in the output dictionary
        output[key] = stacked_tensor

    return output


#data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config = T5Config(n_positions=16384) # Set the maximum token length

model_name = 't5-base' # Load the pre-trained transformer model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=8192)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
# Load the model
model_save_name = 'best_model.pt'
path = f"/kaggle/input/mymodels/{model_save_name}"
#model.load_state_dict(torch.load(path))
model.load_state_dict(torch.load(path, map_location=torch.device(device)))
#/kaggle/input/mymodels/best_model.pt
# Define the training parameters
batch_size = 6
num_epochs = 40
learning_rate = 1e-4

#Learning Rate	Effect
#1e-1	Fast learning rate. May overfit the training data.
#1e-2	Medium learning rate. Good balance between speed and accuracy.
#1e-3	Slow learning rate. May take longer to train, but less likely to overfit.
#1e-4	Very slow learning rate. May take very long to train, but very unlikely to overfit.

# Create data loaders for training, validation, and testing
train_dataset = RelationshipRolesDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = custom_collate)

val_dataset = RelationshipRolesDataset(val_df, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = custom_collate)

test_dataset = RelationshipRolesDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = custom_collate)


# Set the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# Define the number of training steps and warm-up steps
total_steps = len(train_loader) * num_epochs
warmup_steps = int(total_steps * 0.1)  # 10% of total_steps

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

best_val_loss = float('inf')

# Define the gradient accumulation steps
gradient_accumulation_steps = 3  # adjust this value as needed
max_grad_norm = 2.0  # Define the maximum gradient norm for gradient clipping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Early stopping based on validation loss
early_stopping_patience=3
consecutive_non_improvement = 0  # Counter for consecutive epochs with no improvement
for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_attention_mask = batch['target_attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            decoder_attention_mask=target_attention_mask
        )
        loss = outputs.loss
        total_loss += loss.item()

        # Perform gradient accumulation if needed
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update model parameters every gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Clearing GPU and CPU memory cache
            torch.cuda.empty_cache()
            gc.collect()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch+1}, Train Loss: {average_loss:.4f}')
    
    # Clearing GPU and CPU memory cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Validation
    model.eval()
    total_val_loss = 0
    
    # Initialize variables to track accuracy
    val_correct = 0
    val_total = 0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids,
                decoder_attention_mask=target_attention_mask
            )
            loss = outputs.loss
            total_val_loss += loss.item()
            
            # Calculate validation accuracy
            _, predicted = torch.max(outputs.logits, 2)
            val_total += target_ids.size(0) * target_ids.size(1)
            val_correct += (predicted == target_ids).sum().item()

            # Append predictions and targets for evaluation
            val_predictions.extend(predicted.view(-1).tolist())
            val_targets.extend(target_ids.view(-1).tolist())

    # Calculate and print validation metrics
    val_accuracy = (val_correct / val_total) * 100
    average_val_loss = total_val_loss / len(val_loader)
    val_precision = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
    val_f1 = f1_score(val_targets, val_predictions, average='weighted', zero_division=0)

    print(f'Epoch: {epoch+1}, Validation Loss: {average_val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Validation Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1-score: {val_f1:.4f}')
    
    # Early stopping
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        consecutive_non_improvement = 0  # Reset the counter
        torch.save(model.state_dict(), './best_model.pt')
    else:
        consecutive_non_improvement += 1
        if consecutive_non_improvement >= early_stopping_patience:
            print("Validation loss did not improve for a while. Stopping early.")
            break  # Break the loop if patience is exceeded
    # Clearing GPU and CPU memory cache
    torch.cuda.empty_cache()
    gc.collect()
print(best_val_loss)

# Load the best model for testing
model.load_state_dict(torch.load('best_model.pt'))
# Testing
model.eval()
total_test_loss = 0

# Initialize variables to track accuracy
test_correct = 0
test_total = 0
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_attention_mask = batch['target_attention_mask'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
            decoder_attention_mask=target_attention_mask
        )
        loss = outputs.loss
        total_test_loss += loss.item()
        
        # Calculate test accuracy
        _, predicted = torch.max(outputs.logits, 2)
        test_total += target_ids.size(0) * target_ids.size(1)
        test_correct += (predicted == target_ids).sum().item()
        
        # Append predictions and targets for evaluation
        test_predictions.extend(predicted.view(-1).tolist())
        test_targets.extend(target_ids.view(-1).tolist())

# Calculate and print test metrics
test_accuracy = (test_correct / test_total) * 100
average_test_loss = total_test_loss / len(test_loader)
test_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)

print(f'Test Loss: {average_test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1-score: {test_f1:.4f}')

torch.cuda.device_count()
