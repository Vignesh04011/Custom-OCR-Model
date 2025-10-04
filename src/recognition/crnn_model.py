import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CRNN for handwritten text recognition.
    Input: grayscale line image [B,1,H,W]
    Output: sequence of character logits [T,B,C]
    """
    def __init__(self, imgH=32, nc=1, nclass=80, nh=256):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH has to be multiple of 16"

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=False),
            nn.LSTM(nh*2, nh, bidirectional=True, batch_first=False)
        )

        self.fc = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)  # [B,C,H,W]
        b, c, h, w = conv.size()
        assert h == 1, "the height must be 1 after convs"
        conv = conv.squeeze(2).permute(2,0,1)  # [W,B,C]
        out, _ = self.rnn(conv)
        T,B,H = out.size()
        out = self.fc(out.view(T*B,H))
        out = out.view(T,B,-1)
        return out
