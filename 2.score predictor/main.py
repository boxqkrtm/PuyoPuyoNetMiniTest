import torch
import torch.nn as nn

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(13*6*7, 13*6*7)
        self.fc2 = nn.Linear(13*6*7, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters())

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def readDataset(self, path):
        datasetFile = open(path, "r")
        X = []
        Y = []
        datastring = datasetFile.readlines()
        for i in range(0, len(datastring), 2):
            Xstring = datastring[i]
            Ystring = datastring[i+1]
            tmp = []
            for j in range(13*6):
                for k in range(7):
                    if(k==int(Xstring[j])):
                        tmp.append(1.0)
                    else:
                        tmp.append(0.0)
            X.append(tmp)
            Y.append(float(Ystring))
        return X, Y
    
    def train(self):
        dataset = self.readDataset("train.txt")
        size = len(dataset[0])
        for batch, (X, y) in enumerate(dataset):
            X, y = X.to(device), y.to(device)

            # 예측 오류 계산
            pred = self(X)
            loss = self.loss(pred, y)

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
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
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()