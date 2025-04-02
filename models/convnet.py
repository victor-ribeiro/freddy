from torchvision.datasets import MNIST
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD
import torch.nn.functional as F

from tqdm import tqdm


def load_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class Convnet(nn.Module):

    def __init__(self, num_classes=10):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.softmax(x, dim=1)
        return x


if __name__ == "__main__":

    class Trainer:
        def __init__(self, model, train_loader, test_loader, criterion, optimizer):
            self.model = model
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.hist = []

        def train_epoch(self):
            self.model.train()
            train_loss = 0
            n = len(self.train_loader.dataset)
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            return train_loss / n

        def epoch_acc(self, test_loader):
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return 100 * correct / total

        def test_epoch(self):
            self.model.eval()
            test_loss = 0
            n = len(self.test_loader.dataset)
            with torch.no_grad():
                for img, labl in self.test_loader:
                    outputs = self.model(img)
                    loss = self.criterion(outputs, labl)
                    test_loss += loss.item()
            return test_loss / n

        def fit(self, epochs):
            bar = tqdm(range(epochs), desc="Training")
            for epoch in bar:
                bar.set_description(f"Epoch {epoch + 1}/{epochs}")

                train_loss = self.train_epoch()
                test_loss = self.test_epoch()
                acc = self.epoch_acc(self.test_loader)
                self.hist.append((train_loss, test_loss, acc))
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%"
                )

    def main():
        import matplotlib.pyplot as plt
        import pandas as pd

        batch_size = 64
        epochs = 5
        learning_rate = 0.01

        train_loader, test_loader = load_data(batch_size)
        model = Convnet()
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=learning_rate)

        trainer = Trainer(model, train_loader, test_loader, criterion, optimizer)
        trainer.fit(epochs)
        hist = pd.DataFrame(
            trainer.hist, columns=["train_loss", "test_loss", "accuracy"]
        )
        hist[["train_loss", "test_loss"]].plot()
        plt.xlabel("Epochs")
        plt.ylabel("Loss/Accuracy")
        plt.title("Training History")
        plt.show()

    main()
