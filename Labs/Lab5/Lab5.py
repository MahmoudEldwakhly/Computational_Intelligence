# ===========================
# Double Moon Binary Classification
# Colab-ready Python script
# ===========================

#  Install / import libraries
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim

# Create folder to save plots
os.makedirs('code/plots', exist_ok=True)

# User inputs for number of points
N1 = int(input("Enter number of points in Red class (Class 1): "))
N2 = int(input("Enter number of points in Blue class (Class 0): "))

#  Generate Double Moon dataset
def make_double_moon(N1, N2, d=0.2, r=1.0, seed=42):
    """Create double moon dataset (2 classes)"""
    np.random.seed(seed)
    # Class 1 (top moon)
    theta1 = np.random.uniform(0, np.pi, N1)
    x1 = r*np.cos(theta1) + np.random.normal(0,d,N1)
    y1 = r*np.sin(theta1) + np.random.normal(0,d,N1)
    # Class 0 (bottom moon)
    theta2 = np.random.uniform(0, np.pi, N2)
    x2 = r*np.cos(theta2) + r + np.random.normal(0,d,N2)
    y2 = -r*np.sin(theta2) - d + np.random.normal(0,d,N2)
    X = np.vstack([np.column_stack([x1,y1]), np.column_stack([x2,y2])])
    y = np.hstack([np.ones(N1), np.zeros(N2)])
    return X, y

SEED = 42
X, y = make_double_moon(N1, N2, d=0.2, r=1.0, seed=SEED)

# Visualize dataset
plt.figure(figsize=(6,5))
plt.scatter(X[y==1,0], X[y==1,1], s=15, c='red', label='Class 1 (Red)')
plt.scatter(X[y==0,0], X[y==0,1], s=15, c='blue', label='Class 0 (Blue)')
plt.legend(); plt.title('Double Moon Data'); plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout(); plt.show()

#  Split dataset into train / val / test
TEST_FRAC = 0.2
VAL_FRAC = 0.2
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y)
val_size = VAL_FRAC / (1.0 - TEST_FRAC)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size, random_state=SEED, stratify=y_temp)
print('Train/Val/Test shapes:', X_train.shape, X_val.shape, X_test.shape)

# ===============================
#  Linear Classifier (Logistic Regression)
# ===============================
lin = LogisticRegression(max_iter=500)
lin.fit(X_train, y_train)

lin_val_pred = lin.predict(X_val)
lin_test_pred = lin.predict(X_test)
lin_val_acc = accuracy_score(y_val, lin_val_pred)
lin_test_acc = accuracy_score(y_test, lin_test_pred)

lin_train_loss = log_loss(y_train, lin.predict_proba(X_train)[:,1])
lin_val_loss   = log_loss(y_val, lin.predict_proba(X_val)[:,1])

print(f'Linear Classifier:')
print(f'  Train Loss = {lin_train_loss:.4f}, Val Accuracy = {lin_val_acc:.4f}, Test Accuracy = {lin_test_acc:.4f}')

# Decision boundary for linear classifier
x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = lin.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, zz, levels=[-0.1,0.5,1.1], cmap='bwr', alpha=0.2)
plt.scatter(X[y==1,0], X[y==1,1], s=15, c='red')
plt.scatter(X[y==0,0], X[y==0,1], s=15, c='blue')
plt.title(f'Linear Decision Boundary (val_acc={lin_val_acc:.3f})')
plt.xlabel('x'); plt.ylabel('y'); plt.tight_layout(); plt.show()

# Pseudo-loss curve for linear classifier using SGDClassifier
sgd = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=1, warm_start=True, random_state=SEED)
train_losses, val_losses = [], []
for epoch in range(50):
    sgd.fit(X_train, y_train)
    y_train_prob = sgd.predict_proba(X_train)[:,1]
    y_val_prob = sgd.predict_proba(X_val)[:,1]
    train_losses.append(log_loss(y_train, y_train_prob))
    val_losses.append(log_loss(y_val, y_val_prob))
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Log Loss'); plt.title('Linear Classifier Loss Curve (SGD)')
plt.legend(); plt.tight_layout(); plt.show()

# ===============================
#  MLNN Classifier (PyTorch)
# ===============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X_train, y_train, X_val, y_val, epochs=300, lr=1e-2, hidden=64):
    model = MLP(hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_losses, val_losses = [], []
    best_val, best_state = float('inf'), None
    patience, patience_ctr = 20, 0

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, ytr)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            v = criterion(model(Xva), yva).item()
        train_losses.append(loss.item())
        val_losses.append(v)

        if v < best_val - 1e-4:
            best_val = v
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses

# Train MLNN
mlp, tr_losses, va_losses = train_mlp(X_train, y_train, X_val, y_val, epochs=300, lr=1e-2, hidden=64)

# Plot loss curves
plt.figure(figsize=(6,4))
plt.plot(tr_losses, label='Train'); plt.plot(va_losses, label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('MLNN Loss Curves')
plt.legend(); plt.tight_layout(); plt.show()

# Evaluate MLNN
with torch.no_grad():
    Xva_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    val_probs = torch.sigmoid(mlp(Xva_t)).cpu().numpy()
    test_probs = torch.sigmoid(mlp(Xte_t)).cpu().numpy()

mlp_val_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
mlp_test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
print(f'MLNN Classifier: val_acc={mlp_val_acc:.4f}, test_acc={mlp_test_acc:.4f}')

# Decision boundary for MLNN
grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
with torch.no_grad():
    probs = torch.sigmoid(mlp(grid_t)).cpu().numpy()
zz = (probs >= 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, zz, levels=[-0.1,0.5,1.1], cmap='bwr', alpha=0.2)
plt.scatter(X[y==1,0], X[y==1,1], s=15, c='red')
plt.scatter(X[y==0,0], X[y==0,1], s=15, c='blue')
plt.title(f'MLNN Decision Boundary (val_acc={mlp_val_acc:.3f})')
plt.xlabel('x'); plt.ylabel('y'); plt.tight_layout(); plt.show()

# ===============================
#  Summary Comparison
# ===============================
print('=== Summary Comparison ===')
print(f'Linear Classifier: val_acc={lin_val_acc:.4f}, test_acc={lin_test_acc:.4f}, val_loss={lin_val_loss:.4f}')
print(f'MLNN Classifier:  val_acc={mlp_val_acc:.4f}, test_acc={mlp_test_acc:.4f}')
