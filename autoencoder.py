import torch
from torch import float32, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def in_list(nparr, listx):
    for t in listx:
        if np.array_equal(nparr, t):
            return True
    return False
# Hyper-parameters 
input_size = 560
learning_rate = 0.01

# AtomData dataset generated using data from csv files
class AtomData(torch.utils.data.Dataset):
    def __init__(self,anchor_file,pos_file, neg_file) -> None:
        super().__init__()
        anchor = pd.read_csv(anchor_file)
        positives = pd.read_csv(pos_file)
        negatives = pd.read_csv(neg_file)
        self.data :pd.DataFrame= pd.concat([anchor, positives, negatives])
        self.data = self.data.drop_duplicates().to_numpy()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        sample = torch.from_numpy(self.data[idx]).float()
        return sample

# NeuralNet defines out neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(560, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 20),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 560),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


def generate_auto_model(a_path, p_path, n_path, model_path):
    """ Takes in paths to CSV atoms containing atom data; creates instance of AtomData and splits
    it into test and training sets. Trains an autoencoder on these atoms and saves the model.
    Note: Although the autoencoder does not require triplets, it trains on the same atoms that were
    used in the triplets.

    :param a_path:
    :param p_path:
    :param n_path:
    :param model_path:
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    batch_size = 128
    num_epochs = 200
    #Data prep
    triplet_data = AtomData(a_path, p_path, n_path)
    
    train, test = torch.utils.data.random_split(triplet_data, [int(len(triplet_data)*0.8), len(triplet_data) - int(len(triplet_data)*0.8)])
    loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
    model = NeuralNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    model.train()
    loss_list =[]
    epoch_list = [i+1 for i in range(num_epochs)]
    for epoch in range(num_epochs):
        running_loss =[]
#        for i, (anchor, positive,negative) in enumerate(loader):
        for i, (sample) in enumerate(loader):
#            anchor = anchor.to(device)
#            positive = positive.to(device)
#            negative = negative.to(device)
            sample = sample.to(device)
            optimizer.zero_grad()
            s_out = model(sample)
#            a_out = model(anchor)
#            p_out = model(positive)
#            n_out = model(negative)

#            loss = criterion(a_out,p_out,n_out)
            loss = criterion(sample, s_out)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
    torch.save(model.state_dict(), model_path)

# prepares and splits data into training and testing sets
# creates DataLoader object for training set and initializes a new NeuralNet object to be trained
# trains the NeuralNet, calculates output and loss, stores averages for each epoch, and saves the trained model
if __name__ == "__main__":
    batch_size = 128
    num_epochs = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
#     Data prep
    triplet_data = AtomData("anchor.csv",
                            "pos.csv",
                            "neg.csv")

    #test_cases = AtomTestData("C:/Users/alxto/Desktop/Inference Control via ML/AtomUnifier/test.csv")
    train, test = torch.utils.data.random_split(triplet_data, [int(len(triplet_data)*0.8), len(triplet_data) - int(len(triplet_data)*0.8)])
    loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = True)



    model = NeuralNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    model.train()
    loss_list =[]
    epoch_list = [i+1 for i in range(num_epochs)]
    for epoch in range(num_epochs):
        running_loss =[]
        for i, (sample) in enumerate(loader):
            sample = sample.to(device)
            optimizer.zero_grad()
            s_out = model(sample)

            loss = criterion(sample, s_out) 
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        print(np.mean(running_loss))
        loss_list.append(np.mean(running_loss))
   # torch.save(model.state_dict(), "auto_encoder.pth")
    plt.plot(epoch_list, loss_list, color='red')
    plt.title("Training Loss")
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Average Loss', fontsize=14)
    plt.grid(True)
    plt.show()
    model.load_state_dict(torch.load("auto_encoder.pth", map_location=torch.device('cpu')))
    model.eval()
    loss_vals = []
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset = test, shuffle = False)
        for i, (sample) in enumerate(test_loader):
            sample = sample.to(device)
            s_out = model(sample)
            loss = criterion(sample, s_out) 
            loss_vals.append(loss.cpu().detach().numpy())
        print("test")
        print(np.array(loss_vals).mean())
