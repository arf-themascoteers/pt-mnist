import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.n_labels = 35
        self.net = nn.Sequential(
            nn.Conv2d(1,16, (5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(800, 10)

        )

    def forward(self, x):
        x = self.net(x)
        outputs = x[0]

        print("Outputs")
        print(outputs)

        print("\n\n e^Outputs")
        print(torch.exp(outputs))

        sum_outputs = torch.sum(torch.exp(outputs))
        print("\n\nSum e(Outputs)")
        print(sum_outputs)

        manual_softmax = torch.exp(outputs / sum_outputs)
        print("\n\nManually Calculated Softmax")
        print(manual_softmax)

        print("\n\nActual Softmax")
        print(F.softmax(x, dim=1))

        print("\n\nManually Calculated LogSoftmax")
        print(torch.log(manual_softmax))

        x = F.log_softmax(x, dim=1)
        print("\n\nActual Log Softmax")
        print(x[0])
        return x