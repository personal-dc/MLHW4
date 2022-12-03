import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()
device = "cuda" if torch.cuda.is_available() else "cpu"

train_correct_classes = [0, 0, 0, 0, 0, 0]
train_total_classes = [0, 0, 0, 0, 0, 0]
valid_correct_classes = [0, 0, 0, 0, 0, 0]
valid_total_classes = [0, 0, 0, 0, 0, 0]
test_correct_classes = [0, 0, 0, 0, 0, 0]
test_total_classes = [0, 0, 0, 0, 0, 0]

train_loss_list = []
test_loss_list = []
epoch_list= []

batch_size = 50
epochs = 20

class CSV_dataset(Dataset):
    def __init__(self, filepath): 
        read_csv = pd.read_csv(filepath)
        num_rows = len(read_csv.axes[0])
        num_cols = len(read_csv.axes[1])

        x = read_csv.iloc[0:num_rows, 0:num_cols - 1].values
        y = read_csv.iloc[0:num_rows, num_cols - 1].values
        
        self.y_list = y

        self.X_tensor = torch.tensor(x, dtype = torch.float32)

        self.y_tensor = le.fit_transform(y)

    def __len__(self):
        return len(self.y_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

        
class NeuralNetwork(nn.Module):
    def __init__(self, num_layers, num_neuron_list):
        neuron_layers = [nn.Linear(num_neuron_list[0], num_neuron_list[0]), nn.Identity()]
        for i in range (1, num_layers):
            neuron_layers.append(nn.Linear(num_neuron_list[i - 1], num_neuron_list[i]))
            neuron_layers.append(nn.ReLU())
        super().__init__()
        self.linear_relu_stack = nn.Sequential(*neuron_layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# model = NeuralNetwork(3, [300, 64, 6]).to(device) -- part c
# model = NeuralNetwork(4, [300, 64, 64, 6]).to(device) -- part d
# model = NeuralNetwork(3, [300, 128, 6]).to(device) -- part e
model = NeuralNetwork(4, [300, 128, 128, 6]).to(device) # -- part f

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

train_dataset = CSV_dataset("./hw04_data/train.csv")
test_dataset = CSV_dataset("./hw04_data/test.csv")
val_dataset = CSV_dataset("./hw04_data/validation.csv")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

def data_stats(dataset):
    genre_dict = {}
    genre_list = dataset.y_list
    for genre in genre_list:
        val = genre_dict.get(genre, 0)
        genre_dict[genre] = val+1
    print("number of genres are", len(genre_dict))
    for item in genre_dict.items():
        print(item[0], "has", item[1], "instances")

def add_points(x_list, y_list, clr, label):
    plt.plot(x_list, y_list, color = clr, marker = "o", label = label)
    plt.legend(loc="upper right")

def train(dataloader, model, loss_fn, optimizer, correct_classes, total_classes):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for i in range (len(X)):
            total_classes[y[i]] += 1
            if (pred[i].argmax(0) == y[i]):
                correct_classes[y[i]] += 1

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    train_loss_list.append(loss.item())


def test(dataloader, model, loss_fn, correct_classes, total_classes, run_type):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in range (len(X)):
                total_classes[y[i]] += 1
                if (pred[i].argmax(0) == y[i]):
                    correct_classes[y[i]] += 1

    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    if (run_type == "valid"):
        test_loss_list.append(test_loss)


def print_stats(correct_classes, total_classes):
    inverse_encoded = le.inverse_transform([i for i in range(6)])
    correct = 0
    total = 0
    for i in range(6):
        genre = inverse_encoded[i]
        correct += correct_classes[i]
        total += total_classes[i]
        print("mean accuracy for", genre, "is", str(correct_classes[i]/total_classes[i]))
    print("total mean accuracy is", str(correct/total))

for t in range(epochs):
    epoch_list.append(t+1)
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, train_correct_classes, train_total_classes)
    test(val_dataloader, model, loss_fn, valid_correct_classes, valid_total_classes, "valid")


test(test_dataloader, model, loss_fn, test_correct_classes, test_total_classes, "test")

print('---------train stats---------')
print_stats(train_correct_classes, train_total_classes)
print()
print('---------validation stats---------')
print_stats(valid_correct_classes, valid_total_classes)
print()
print('---------test stats---------')
print_stats(test_correct_classes, test_total_classes)

add_points(epoch_list, train_loss_list, "red", "train")
add_points(epoch_list, test_loss_list, "blue", "test")
plt.show()

print("train dataset stats")
data_stats(train_dataset)
print()
print("validation dataset stats")
data_stats(val_dataset)
print()
print("test dataset stats")
data_stats(test_dataset)

