from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#from datasets import load_dataset, DatasetDict
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import pandas as pd

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # Extract input encodings and labels from the Dataset object
        self.encodings = {
            "input_ids": dataset["input_ids"],
            "attention_mask": dataset["attention_mask"]
        }
        self.labels = dataset["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Access encodings and labels for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
        
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(SentimentClassifier, self).__init__()
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)

        # Define a dropout layer
        self.dropout = nn.Dropout(0.3)

        # Define a linear layer for classification
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        # Use the [CLS] token output
        pooled_output = outputs[1]
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Apply classification layer
        logits = self.classifier(pooled_output)
        return logits
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
current_model = SentimentClassifier("bert-base-uncased", num_labels=2)
current_model.load_state_dict(torch.load("sentiment_model_epoch_3.pt"))
current_model.to(device)

def prepare_dataset_from_csv(csv_path, tokenizer, max_length=256):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Tokenize texts
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    # Create dataset dictionary
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'label': torch.tensor(df['label'].tolist())
    }

    return IMDBDataset(dataset_dict)

def evaluate_metrics(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get predictions
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='binary'
    )
    accuracy = accuracy_score(all_labels, all_predictions)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Usage:
test_dataset = prepare_dataset_from_csv("test_data_movie.csv", tokenizer)
# test_dataset = torch.load("test_dataset.pt")
test_loader = DataLoader(test_dataset, batch_size=16)
metrics = evaluate_metrics(current_model, test_loader, device)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")


# Define the new sequence
sequence = """Julie Andrews satirically prods her own goody-two-shoes image in this overproduced musical comedy-drama, but if she approaches her role with aplomb, she's alone in doing so. Blake Edwards' film about a woman who is both music-hall entertainer and German spy during WWI doesn't know what tone to aim for, and Rock Hudson has the thankless task of playing romantic second-fiddle. Musicals had grown out of favor by 1970, and elephantine productions like "Star!" and this film really tarnished Andrews' reputation, leaving a lot of dead space in her catalogue until "The Tamarind Seed" came along. I've always thought Julie Andrews would've made a great villain or shady lady; her strong voice could really command attention, and she hits some low notes that can either be imposing or seductive. Husband/director Edwards seems to realize this, but neither he nor Julie can work up much energy within this scenario. Screenwriter William Peter Blatty isn't a good partner for Edwards, and neither man has his heart in this material. Beatty's script offers Andrews just one fabulous sequence--a striptease. *1/2 from ****"""

# Preprocess the sequence
inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=256)

# Move tensors to the device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Run the model in evaluation mode
current_model.eval()
with torch.no_grad():
    logits = current_model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=1)  # Get the predicted class (0 or 1)

# Decode the prediction
label_map = {0: "Negative", 1: "Positive"}  # Adjust based on your dataset's labels
predicted_label = label_map[predictions.item()]
print(f"Predicted Sentiment: {predicted_label}")