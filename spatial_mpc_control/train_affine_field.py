import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

# The Continuous Piecewise Affine (CPWA) Network
class ExplicitMPCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # 4 actuator outputs
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate():
    # 1. Load Data
    try:
        with open('mpc_expert_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Data not found. Run generate_mpc_dataset.py first.")
        return

    X = torch.tensor(data['X'], dtype=torch.float32)
    U = torch.tensor(data['U'], dtype=torch.float32)

    # 2. Train/Test Split (90/10)
    split_idx = int(len(X) * 0.9)
    X_train, U_train = X[:split_idx], U[:split_idx]
    X_test, U_test = X[split_idx:], U[split_idx:]
    
    model = ExplicitMPCPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    print("Training Affine Field Network...")
    losses = []
    
    # 3. Training Loop
    for epoch in range(1500):
        optimizer.zero_grad()
        pred_U = model(X_train)
        loss = loss_fn(pred_U, U_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch} | MSE Loss: {loss.item():.4f}")

    # 4. Save the Weights for ROS 2 Deployment
    torch.save(model.state_dict(), 'affine_policy.pth')
    print("Model weights saved to 'affine_policy.pth'")

    # 5. Evaluate on Unseen Test Data
    model.eval()
    with torch.no_grad():
        U_pred = model(X_test)
        test_loss = loss_fn(U_pred, U_test)
    print(f"Validation MSE: {test_loss.item():.4f}")

    # 6. Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(losses, color='purple', linewidth=2)
    axs[0].set_title("Training Loss")
    axs[0].grid(True)

    axs[1].scatter(U_test.numpy().flatten(), U_pred.numpy().flatten(), alpha=0.3, color='teal')
    axs[1].plot([-300, 300], [-300, 300], color='red', linestyle='--')
    axs[1].set_title("Predicted vs Actual (Test Set)")
    axs[1].grid(True)
    
    plt.savefig('weights_evaluation.png')
    print("Evaluation plot saved to 'weights_evaluation.png'")

if __name__ == "__main__":
    train_and_evaluate()