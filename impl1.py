from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset

# Custom Dataset Class for DataLoader
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.input_ids = torch.stack(dataframe['input_ids'].tolist())
        self.attention_mask = torch.stack(dataframe['attention_mask'].tolist())
        self.labels = torch.tensor(dataframe['label_column'].values)  # Adjust the label column name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

def train_and_evaluate(train_loader, val_loader, hidden_units, dropout_rate, learning_rate, epochs):
    model = SentimentModel(hidden_units, dropout_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_score = 0.0
    for epoch in range(epochs):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                preds = model(input_ids, attention_mask)
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        if val_f1 > best_val_score:
            best_val_score = val_f1

    return best_val_score

# Define Hyperparameters for GridSearch
param_grid = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': [0.3, 0.5, 0.7],
    'learning_rate': [1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20]
}

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare data
X = data[['input_ids', 'attention_mask']]
y = data['label_column']  # Adjust the label column name

best_params = None
best_score = 0.0

# Grid Search to find the best hyperparameters
for hidden_units in param_grid['hidden_units']:
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    fold_scores = []
                    for train_idx, val_idx in kfold.split(X):
                        train_data = data.iloc[train_idx]
                        val_data = data.iloc[val_idx]

                        train_dataset = SentimentDataset(train_data)
                        val_dataset = SentimentDataset(val_data)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        score = train_and_evaluate(train_loader, val_loader, hidden_units, dropout_rate, learning_rate, epochs)
                        fold_scores.append(score)

                    avg_score = np.mean(fold_scores)
                    print(f"Params: HU={hidden_units}, DR={dropout_rate}, LR={learning_rate}, BS={batch_size}, EP={epochs} => Avg F1: {avg_score}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'hidden_units': hidden_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }


