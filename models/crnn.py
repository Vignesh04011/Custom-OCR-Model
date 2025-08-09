import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        """
        img_height     = Height of input image (e.g., 32)
        num_channels   = 1 for grayscale, 3 for RGB
        num_classes    = Number of characters + 1 (for CTC blank token)
        """
        super(CRNN, self).__init__()

        # 🔸 CNN Layers to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # downsample height only

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # preserve width resolution
        )

        # 🔸 Recurrent Layers (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=256, num_layers=2,
            bidirectional=True, batch_first=False
        )

        # 🔸 Final classification layer: maps to num_classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Input:
            x shape: (B, C, H, W)
        Output:
            logits shape: (T, B, num_classes) for CTC loss
        """
        conv_output = self.cnn(x)  # (B, 512, H', W')

        # Collapse height to 1 and prepare for RNN
        batch, channels, height, width = conv_output.size()
        assert height == 1, "Expected height=1 after CNN, got height={}".format(height)
        conv_output = conv_output.squeeze(2)         # (B, 512, W)
        conv_output = conv_output.permute(2, 0, 1)   # (W, B, 512) → (T, B, features)

        # RNN
        rnn_output, _ = self.rnn(conv_output)        # (T, B, 512)
        logits = self.fc(rnn_output)                 # (T, B, num_classes)

        return logits
