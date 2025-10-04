import torch
import torch.nn as nn
import torch.nn.functional as F

class MarkerClassifier(nn.Module):
    """
    Small CNN to classify whether a patch is a question marker (e.g., 'Q1', '1.', '(1)').
    """
    def __init__(self, num_classes=2):
        super(MarkerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def classify_marker(model, patch, device="cpu"):
    """
    Run classification on a small cropped patch.
    """
    model.eval()
    with torch.no_grad():
        patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(patch)
        probs = torch.softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).item(), probs.cpu().numpy()
