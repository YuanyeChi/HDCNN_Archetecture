import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt
import numpy as np

"""
Neuro network structure:
Layer1: 3*3*64 filter with stride of 1, padding of 1 and max pooling of 2*2.
Layer2: 3*3*128 filter with stride of 1, padding of 1 and max pooling of 2*2.
Layer3: 3*3*256 filter with stride of 1, padding of 1 and max pooling of 2*2.

Fully connected layer: 256*16*16 -> 256,1 -> 4 (items of class)
"""
torch.manual_seed(1)


class NeuroNetwork(nn.Module):
    def __init__(self):
        super(NeuroNetwork, self).__init__()
        first_layer = 64
        second_layer = 128
        third_layer = 256
        forth_layer = 512
        fc_layer = 1024
        output = 25
        self.cnn_layers = nn.Sequential(
            # First layer
            nn.Conv2d(in_channels=3,  # RGB 3 layer3
                      out_channels=first_layer,  # Output layer -- the number of filters
                      kernel_size=3,  # Size of filter --3*3
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(first_layer),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second layer
            nn.Conv2d(first_layer, second_layer, 3, 1, 1),
            nn.BatchNorm2d(second_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third layer
            nn.Conv2d(second_layer, third_layer, 3, 1, 1),
            nn.BatchNorm2d(third_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Forth layer
            nn.Conv2d(third_layer, forth_layer, 3, 1, 1),
            nn.BatchNorm2d(forth_layer),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(forth_layer * 8 * 8, fc_layer),
            nn.BatchNorm1d(fc_layer),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_layer, output)
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layers(x)
        return x


class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]


def get_pseudo_labels(dataset, model, threshold=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    idx = []
    labels = []

    for i, batch in enumerate(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * batch_size + j)
                labels.append(int(torch.argmax(x)))

    model.train()
    print("\nNew data: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels)
    return dataset


"""
Pre-process the raw data:

1. Do data augmentation
2.
by resize to 128*128
"""

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    transforms.RandomRotation(degrees=25),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
train_set = DatasetFolder("arcDataset", loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg",
                          transform=train_tfm)
test_set = DatasetFolder("arcValidset", loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg",
                         transform=test_tfm)
add_pseudo_data = False

if add_pseudo_data:
    pseudo_set = DatasetFolder("dataset/arcPseudoset", loader=lambda x: Image.open(x).convert("RGB"), extensions="jpg",
                               transform=train_tfm)
# Batch size of 128
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
# CrossEntropy loss are applied
cross_entropy = nn.CrossEntropyLoss()

cnn = NeuroNetwork().to(device)
cnn.device = device
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-5)

train_loss_record = []
valid_loss_record = []
train_acc_record = []
valid_acc_record = []
# Train for 20 times rounds
n_epochs = 150
best_acc = 0.0

for epoch in range(n_epochs):
    print("Epoch: ", epoch)

    if add_pseudo_data and best_acc > 0.01 and epoch % 1 == 0:
        pseudo_set = get_pseudo_labels(pseudo_set, cnn)
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                  drop_last=True)

    cnn.train()

    train_loss = []
    train_acc = []

    for batch in train_loader:
        data, labels = batch

        predict = cnn(data.to(device))
        loss = cross_entropy(predict, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.tensor(predict.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    cnn.eval()
    test_loss = []
    test_acc = []
    for batch in test_loader:
        data, labels = batch

        with torch.no_grad():
            predict = cnn(data.to(device))

        loss = cross_entropy(predict, labels.to(device))

        acc = torch.tensor(predict.argmax(dim=-1) == labels.to(device)).float().mean()
        test_loss.append(loss.item())
        test_acc.append(acc)

    valid_loss = sum(test_loss) / len(test_loss)
    valid_acc = sum(test_acc) / len(test_acc)

    print(f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    if valid_acc > best_acc:
        best_acc = valid_acc

    train_loss_record.append(train_loss)
    valid_loss_record.append(valid_loss)
    train_acc_record.append(train_acc)
    valid_acc_record.append(valid_acc)

x = np.arange(len(train_acc_record))
plt.plot(x, train_acc_record, color="blue", label="Train")
plt.plot(x, valid_acc_record, color="red", label="Valid")
plt.legend(loc="upper right")
plt.show()

x = np.arange(len(train_loss_record))
plt.plot(x, train_loss_record, color="blue", label="Train")
plt.plot(x, valid_loss_record, color="red", label="Valid")
plt.legend(loc="upper right")
plt.show()
