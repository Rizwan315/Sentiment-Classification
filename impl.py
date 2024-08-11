from sklearn.model_selection import KFold, GridSearchCV
import numpy as np

# Hyperparameters
hidden_units = 128
dropout_rate = 0.5
learning_rate = 1e-4
batch_size = 32
epochs = 20

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate(train_idx, val_idx):
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
    
    train_loader = DataLoader(SentimentDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_data), batch_size=batch_size, shuffle=False)
    
    model = SentimentModel(hidden_units, dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        # Validation code here
    
    return validation_metrics  # e.g., accuracy, F1-score

# Perform K-Fold Cross-Validation
results = []
for train_idx, val_idx in kfold.split(data):
    metrics = train_and_evaluate(train_idx, val_idx)
    results.append(metrics)

# Average results
avg_result = np.mean(results, axis=0)
print(f"Average results across folds: {avg_result}")
