import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Tambahan untuk log training
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    def train_model(self, X_train, y_train, val_data=None, epochs=100, batch_size=64, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            self.train()
            permutation = torch.randperm(X_train.size()[0])
            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = self(batch_x).squeeze()
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            train_acc = correct / total
            self.train_losses.append(total_loss / (total / batch_size))
            self.train_accs.append(train_acc)

            # Validation
            if val_data is not None:
                self.eval()
                X_val, y_val = val_data
                with torch.no_grad():
                    val_outputs = self(X_val).squeeze()
                    val_loss = loss_fn(val_outputs, y_val).item()
                    val_pred = (val_outputs > 0.5).float()
                    val_acc = (val_pred == y_val).sum().item() / y_val.size(0)

                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {self.train_losses[-1]:.4f}, Acc: {train_acc:.4f}" + (
                f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_data else "")
            )
