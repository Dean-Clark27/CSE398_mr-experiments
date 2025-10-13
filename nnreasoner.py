# from sqlalchemy import true
from random import shuffle
import torch
from torch import Tensor, nn
from scipy.signal import savgol_filter

# from torchvision import datasets
# from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

# from torchmetrics.functional import accuracy, f1_score, fbeta_score
# from sklearn.model_selection import train_test_split
import argparse
from kbencoder import (
    generate_autoencoder_embeddings,
    generate_chainbased_embeddings,
    generate_termwalk_embeddings,
    generate_unification_embeddings,
)

from kbparser import parse_atom, parse_rule
from vocab import Vocabulary
from torch.utils.data import random_split, DataLoader, Dataset

# Hyper-parameters
# input_size = 60
# input_size = 90774
hidden_size1 = 30
hidden_size2 = 15
num_classes = 1
# num_epochs = 500 # make sure to print epoch number to keep track of timing
batch_size = 25
learning_rate = 0.05

LOSS_STEP = 50


def in_list(nparr, listx):
    for t in listx:
        if np.array_equal(nparr, t):
            return True
    return False


# takes in positive and negative examples and returns tuple of input samples and their labels
# Note: the input data can't be sparse, because those format do not support subscripting, etc.
class ReasonerData(Dataset):
    def __init__(self, data, device="cpu") -> None:
        super().__init__()
        self.data = (
            torch.index_select(
                data, 1, torch.tensor(range(data.shape[1] - 1)).to(device)
            )
            .cpu()
            .to_dense()
            .numpy()
        )
        # print(data.dtype)
        self.labels = (
            torch.index_select(data, 1, torch.tensor(
                [data.shape[1] - 1]).to(device))
            .cpu()
            .to_dense()
            .numpy()
        )
        # print(data.dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        sample = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float().reshape(-1)
        return sample, label


# defines the architecture of the neural network with 2 hidden layers, as well as the input and output layers
class NeuralNet(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.LazyLinear(hidden_size1)
        self.sig1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size1, num_classes)
        self.sig2 = nn.Sigmoid()
        # TODO: Not using l3 and sig3?
        self.l3 = nn.Linear(hidden_size2, num_classes)
        self.sig3 = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.sig1(out)
        out = self.l2(out)
        out = self.sig2(out)
        # out = self.l3(out)
        # out= self.sig3(out)
        return out


def get_score(embedding: torch.Tensor, model: NeuralNet):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        # embedding = torch.from_numpy(embedding).to(device)
        score = model(embedding)
        return score.item()


def train_reasoning_model(
        training_file, num_epochs, save_file, vocab: Vocabulary,
        embed_type="unification", embed_path="rKB_model.pth"
):
    print("Training " + embed_type)
    print(f"Embed size: {embed_size}")

    # embeddings = torch.load(args.embeddings).cpu().to_dense()   # what if using CUDA system?

    # embeddings = pd.DataFrame(embeddings)
    # Note: I think Sparse should only be used for chain-based and term-walk
    # embeddings = embeddings.astype(pd.SparseDtype("float", 0.0))

    # only use CPU for massaging data (important for termwalk)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading examples from " + training_file)

    examples = []
    with open(training_file, mode="r") as f:
        lines = f.readlines()[1:-1]
        lines = [x.lstrip() for x in lines if x.strip() and x[0] != "%"]

        print("Parsing examples...")

        for line in lines:
            goal, rule, score = map(str.strip, line.split("\t"))
            examples.append([parse_atom(goal), parse_rule(rule), float(score)])

    print("Generating embeddings...")

    # TODO: update this to use EmbedModel class (combines the embed size and path)
    embeddings_func: dict = {
        # TDO: this is where we can pass in a different embedding model file
        "unification": lambda ex, device: generate_unification_embeddings(
            ex, device, vocab, embed_size, embed_path
        ),
        "autoencoder": lambda ex, device: generate_autoencoder_embeddings(
            ex, device, vocab
        ),
        "chainbased": lambda ex, device: generate_chainbased_embeddings(
            ex, device, embed_size
        ),
        "termwalk": lambda ex, device: generate_termwalk_embeddings(
            ex, device, vocab
        ),
    }
    embeddings = embeddings_func.get(
        embed_type, "unification")(examples, device.type)

    data = ReasonerData(embeddings, device.type)
    data = random_split(data, [0.9, 0.1])  # TODO: why are we doing this split?
    train_loader = DataLoader(
        dataset=data[0], batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=data[1], batch_size=batch_size, shuffle=True)

    print("Loaded data...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    criterion = torch.nn.BCELoss()
    model = NeuralNet(hidden_size1, hidden_size2, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []
    # best_test_loss = float('inf')
    # best_model_state = None
    epoch = 0

    # training
    while True:
        running_loss = []

        model.train()
        for (sample, label) in train_loader:
            sample = sample.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            s_out = model(sample)

            loss = criterion(s_out, label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        loss_list.append(np.mean(running_loss))

        # # Validation loop
        # model.eval()
        # with torch.no_grad():
        #     test_losses = []
        #     for sample, label in test_loader:
        #         sample = sample.to(device)
        #         label = label.to(device)

        #         output = model(sample)
        #         test_loss = criterion(output, label)
        #         test_losses.append(test_loss.item())

        #     # Calculate average validation loss for the epoch
        #     epoch_test_loss = np.mean(test_losses)

        #     print(
        #         f"\r{epoch}\t{str(loss_list[-1])[:7]} ({str(epoch_test_loss)[:7]})", end="")
        #     if epoch % LOSS_STEP == 0:
        #         print()

        # epoch += 1

        # # Check for early stopping
        # if epoch_test_loss < best_test_loss:
        #     best_test_loss = epoch_test_loss
        #     best_model_state = model.state_dict()
        # else:
        #         break

        primary_smoothing_window = 50
        secondary_smoothing_window = 5
        if len(loss_list) > primary_smoothing_window:
            smoothed_data = savgol_filter(
                loss_list, primary_smoothing_window, 3)
            current_gradient = np.mean(np.gradient(smoothed_data)
                                       [-1::-secondary_smoothing_window])
        print(
            f"\r{epoch}\t{str(loss_list[-1])[:7]} ({str(current_gradient)[:7] if len(loss_list) > primary_smoothing_window else '-'})", end="")
        if epoch % LOSS_STEP == 0:
            print()

        epoch += 1

        if epoch >= 750 and epoch % LOSS_STEP == 0:
            max_gradient = -0.00015
            max_epochs = 1500
            if current_gradient > max_gradient:
                print()
                break

    epoch_list = [i + 1 for i in range(epoch)]

    if save_file is not None:
        torch.save(model.state_dict(), args.save_model)  # rule_classifier.pth

    plt.plot(epoch_list, loss_list, color="red")
    plt.title(
        f"Guided Training Loss - {str(loss_list[-1])[:7]} (p:{len(vocab.predicates)}, c:{len(vocab.constants)}, a:{vocab.maxArity}, e:{embed_size})")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.grid(True)
    # plt.show()
    plt.savefig(
        f"guided_loss-{embed_type}{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.png")
    print("Saved training loss figure.")

    # model.load_state_dict(torch.load("rule_classifier.pth"))
    # model.eval()
    # with torch.no_grad():
    #     test_loader = torch.utils.data.DataLoader(dataset = test, shuffle = False)
    #     count = 0
    #     true_labels = []
    #     predicted_labels = []
    #     for i, (sample,label) in enumerate(test_loader):
    #         sample = sample.to(device)
    #         label = label.to(device)
    #         s_out = model(sample)
    #         print(sample.cpu().detach().numpy().tolist())
    #         true_labels.append(label.item())
    #         if s_out > 0.5:
    #             if(label.item() == 1):
    #                 print("correct")

    #             predicted_labels.append(1)
    #         else:
    #             if(label.item() == 1):
    #                 print("false")

    #             predicted_labels.append(0)

    #     true_labels = torch.tensor(true_labels).type(torch.IntTensor)
    #     predicted_labels = torch.tensor(predicted_labels)
    #     print(f"F0.5 score:{fbeta_score(predicted_labels,true_labels,beta = 0.5, num_classes=2)}")
    #     print(f"F2 score:{fbeta_score(predicted_labels,true_labels, beta = 2,num_classes=2)}")
    #     print(f"F1 score:{f1_score(predicted_labels,true_labels, num_classes=2)}")
    #     print(f"Accuracy score:{accuracy(predicted_labels,true_labels)}")


# trains neural network, recording and plotting the training loss of each epoch
# trained model parameters saves
if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "-t",
        "--training_file",
        default="mr_train_examples.csv",
        help="File path for the training data (goal/rule/score)"
    )
    aparser.add_argument(
        "-s",
        "--save_model",
        default="mr_model.pt",
        help="File path to save the trained model. Defaults to mr_model.pt."
    )
    aparser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train. Default: 1000"
    )
    aparser.add_argument(
        "--embed_type",
        choices=["unification", "autoencoder", "chainbased", "termwalk"],
        default="unification",
        help="Type of embedding",
    )
    aparser.add_argument(
        "--vocab_file", default="vocab", help="Path to dave generated vocab to."
    )
    aparser.add_argument("-e", "--embed_size", type=int, default=50,
                         help="Embed size. Defaults to 50")
    aparser.add_argument("--embed_model_path", default="rKB_model.pth",
                         help="Path to read a trained embedding model from")
    aparser.add_argument("--hash_const",  action="store_true",
                     help="Use hash-based constant encoding instead of one-hot.")
    args = aparser.parse_args()

    vocab = Vocabulary()
    vocab.init_from_vocab(args.vocab_file)
    embed_size = args.embed_size
    vocab.set_use_hash_const(args.hash_const)

    print("States from vocab: " + args.vocab_file)
    vocab.print_summary()
    print()

    default_save_files = {
        "unification": "uni_mr_model.pt",
        "autoencodeer": "auto_mr_model.pt",
        "chainbased": "cb_mr_model.pt",
        "termwalk": "tw_mr_model.pt"
    }

    if args.save_model == "mr_model.pt":
        args.save_model = default_save_files[args.embed_type]

    train_reasoning_model(
         args.training_file, args.num_epochs, args.save_model, vocab, args.embed_type, args.embed_model_path
    )
