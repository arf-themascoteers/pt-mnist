from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

def train():
    NUM_EPOCHS = 3
    BATCH_SIZE = 1000

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)

            actual_output = y_true[0].item()
            print("\n\nActual Output")
            print(actual_output)

            confidence_for_actual_output = y_pred[0][actual_output].item()
            print(f"\n\nConfidence for {actual_output}")
            print(confidence_for_actual_output)

            print("\n\nCalculated Loss")
            print(-confidence_for_actual_output)

            loss = F.nll_loss(y_pred, y_true)
            print("\n\nActual Loss")
            print(loss)
            exit(0)

            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    torch.save(model.state_dict(), 'models/cnn.h5')
    return model

train()
exit(0)


