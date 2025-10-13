import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse
from atomgenerator import create_unity_embeddings
from prints import print_progress_bar
from vocab import Vocabulary

def in_list(nparr, listx):
    for t in listx:
        if np.array_equal(nparr, t):
            return True
    return False


HARDEST_EXAMPLES = False         # set the default to false, because True makes training exceedingly long
# Hyper-parameters
hidden_size1 = 256
hidden_size2 = 128
# num_classes = 300
# change to number of facts
# preds * (consts ^ max arity)
num_epochs = 500
batch_size = 64
learning_rate = 0.00001
margin = 1.0
recalibrate_epochs = 10
patience = recalibrate_epochs * 2 # If two recalibrations dont lead to a lower loss, just stop the program

# loads data into csv files and splits it into training and testing sets
class AtomData(torch.utils.data.Dataset):
    def __init__(self, anchor_file, pos_file, neg_file) -> None:
        super().__init__()
        # anchor = np.array(anchor_file)
        # positives = np.array(pos_file)
        # negatives = np.array(neg_file)
        
        #YW: Used for getting similarity directly from anchor, positive, negative one hot encoding files.
        # anchor[anchor == ''] = 0.0
        # anchor = anchor.astype(np.float64)

        # positives[positives == ''] = 0.0
        # positives = positives.astype(np.float64)

        # negatives[negatives == ''] = 0.0
        # negatives = negatives.astype(np.float64)

        self.anchor = np.array(anchor_file)
        self.positives = np.array(pos_file)
        self.negatives = np.array(neg_file)

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        anchor_sample = torch.from_numpy(self.anchor[idx]).float()
        pos_sample = torch.from_numpy(self.positives[idx]).float()
        neg_sample = torch.from_numpy(self.negatives[idx]).float()
        return anchor_sample, pos_sample, neg_sample

class AtomDataWithInd(AtomData):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        anchor_sample = torch.from_numpy(self.anchor[idx]).float()
        pos_sample = torch.from_numpy(self.positives[idx]).float()
        neg_sample = torch.from_numpy(self.negatives[idx]).float()
        return anchor_sample, pos_sample, neg_sample, idx

# 3 layered neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)

def merge(left, right):
    sorted_array = []
    left_index, right_index = 0, 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index][0] <= right[right_index][0]:
            sorted_array.append(left[left_index])
            left_index += 1
        else:
            sorted_array.append(right[right_index])
            right_index += 1

    sorted_array.extend(left[left_index:])
    sorted_array.extend(right[right_index:])

    return sorted_array

def generate_unification_model(a_path, p_path, n_path,
        model_path, 
        vocab: Vocabulary, 
        embed_size: int, 
        save_unity_embeddings=False, 
        num_triplets = 70000, 
        use_triplet_file: str = False, 
        use_legacy_embeddings=False, 
        triplet_set_size = 3,
        use_hash_const=False):
    """Trains an embedding model using triplet loss with atoms that unify.
    Takes paths to anchor, pos, and neg csvs, saving the trained model to its path

    :param a_path:
    :param p_path:
    :param n_path:
    :param model_path:
    :param use_triplet_file: Path to triplet file to use
    :return: None
    """
    #TODO: Make a custom NN for the trips with indexing, so you can do the hard manipulations

    # JDH 7/16/25: commented out because not sure where this is being used. vocab.hashBits does not exist
    # if use_legacy_embeddings:
    #     input_size = len(vocab.predicates) + (
    #         (len(vocab.variables) + len(vocab.constants)) * vocab.maxArity
    #     )
    # else:
    #     input_size = vocab.hashBits

    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    device = get_device()
    print(f"Using {device} device")
    print(f"Uni model embed size: {embed_size}")
    # Data prep

    anchor_list, positive_list, negative_list = create_unity_embeddings(
        vocab, a_path, p_path, n_path, num_triplets, save_unity_embeddings, use_triplet_file, use_legacy_embeddings, triplet_set_size, use_hash_const)

    print("\rConverting to Dataset...", end="\r")
    triplet_data = AtomDataWithInd(anchor_list, positive_list, negative_list)

    train_size = int(0.8 * len(triplet_data))
    val_size = int(0.2 * len(triplet_data))
    if (train_size + val_size) != len(triplet_data):
        train_size += 1
    train, val = torch.utils.data.random_split(triplet_data, [train_size, val_size])

    print("\rLoading Train...", end="\033[K\r")
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    print("\rLoading Validation...", end="\033[K\r")
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)

    # this needs to be after the call to create_unity_embeddings, to ensure a consistent input size
    input_size = vocab.get_input_encode_size()
    print("Input encode size = ", input_size)

    model = NeuralNet(input_size, hidden_size1,
                      hidden_size2, embed_size).to(device)
    criterion = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance(),
        margin=margin,
        reduction="mean",
    )
    # switched from SGD to Adam for testing
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training with early stopping
    best_loss = float('inf')
    patience_counter = 0
    loss_list = []
    val_loss_list = []
    epoch_list = [i + 1 for i in range(num_epochs)]
    for epoch in range(num_epochs): 
        model.train()
        running_loss = []
        hard_examples = []
        dist = torch.nn.PairwiseDistance()

        # Reset to regular size before training epoch
        if HARDEST_EXAMPLES and epoch != 0 and epoch % recalibrate_epochs == 0:
            train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            model.eval()
        
        for (anchor, positive, negative, batch_idx) in train_loader:
            if HARDEST_EXAMPLES and epoch != 0 and epoch % recalibrate_epochs == 0:
                print(f"\rGetting hard examples...", end='\033[K\r')
                for (a, p, n, train_data_idx) in zip(anchor, positive, negative, batch_idx):
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    a_flat = model(a.unsqueeze(0))
                    p_flat = model(p.unsqueeze(0))
                    n_flat = model(n.unsqueeze(0))

                    loss = criterion(a_flat, p_flat, n_flat)
                    hard_examples.append([loss.cpu().detach().numpy(), train_data_idx.item()])
            else:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad()
                a_out = model(anchor)
                p_out = model(positive)
                n_out = model(negative)

                loss = criterion(a_out, p_out, n_out)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.cpu().detach().numpy())

        train_loss = np.mean(running_loss)
        loss_list.append(train_loss)

        model.eval()
        # Force model to learn on hard examples
        if HARDEST_EXAMPLES and epoch != 0 and epoch % recalibrate_epochs == 0:
            print("\rSorting computations...", end='\033[K\r')
            hard_examples = merge_sort(hard_examples)[int(len(hard_examples)/2):]
            print("\rUpdating train data...", end='\033[K\r')
            combined_set = [item[1] for item in hard_examples] # removes similarity scores, only indices
            combined_set = list(set(combined_set)) # removes duplicates
            
            new_train_set = torch.utils.data.Subset(triplet_data, indices=combined_set)
            train_loader = DataLoader(dataset=new_train_set, batch_size=batch_size, shuffle=False)
            # TODO: Maybe delay recalibration if validation loss drops significantly?

        # validation step
        val_running_loss = []
        with torch.no_grad():
            for anchor, positive, negative, idx in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                a_out = model(anchor)
                p_out = model(positive)
                n_out = model(negative)

                val_loss = criterion(a_out, p_out, n_out)
                val_running_loss.append(val_loss.cpu().numpy())
        val_loss = np.mean(val_running_loss)
        val_loss_list.append(val_loss)

        print_progress_bar(epoch, num_epochs - 1, suffix="Epoch", shown='percent')
        print(f" | [Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}]", end='\r')

        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)  # save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Saved model.")

    # to make sure plots are same dimension
    min_length = min(len(epoch_list), len(loss_list))
    epoch_list = epoch_list[:min_length]
    loss_list = loss_list[:min_length]

    plt.figure()
    plt.plot(epoch_list[:len(loss_list)], loss_list, label='Training Loss', color='red')
    plt.plot(epoch_list[:len(val_loss_list)], val_loss_list, label='Validation Loss', color='blue')
    plt.title(
                f"Unification and Validation Training Loss (p:{len(vocab.predicates)}, c:{len(vocab.constants)}, "
                f"a:{vocab.maxArity}, e:{embed_size})")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"training_and_validation_loss.png")
    plt.close()
    #
    plt.figure()
    plt.plot(epoch_list, loss_list, color="red")
    plt.title(
        f"Unification Training Loss (p:{len(vocab.predicates)}, c:{len(vocab.constants)}, a:{vocab.maxArity}, e:{embed_size})")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.grid(True)
    # plt.show()
    plt.savefig(
        f"training_loss-{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.png")
    plt.close()
    print("Saved plot.")

# prepares, splits, and trains the data for a specified number of epochs. The training loss is recorded.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the \
    unifier model, which generates embeddings for atoms."
    )
    parser.add_argument(
        "-s",
        "--save_model",
        help="Save the trained model to \
    the provided path. If no path is given, the model will not be saved.",
    )
    parser.add_argument(
        "-l",
        "--load_model",
        help="Load the model from the \
    given path. If no path is given, the trained model will be used.",
    )
    parser.add_argument(
        "--vocab_file", default="vocab", help="Path to save generated vocab to."
    )
    parser.add_argument("-e", "--embed_size", default=20,
                        help="Embed size. Defaults to 20")

    print("Deprecated code")
    args = parser.parse_args()
    vocab = Vocabulary()
    vocab.init_from_vocab(args.vocab_file)
    # num_classes = get_embed_size(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    #     Data prep
    triplet_data = AtomData(
        "train_anchors.csv", "train_positives.csv", "train_negatives.csv"
    )

    # test_cases = AtomTestData("C:/Users/alxto/Desktop/Inference Control via ML/AtomUnifier/test.csv")
    train, test = torch.utils.data.random_split(
        triplet_data,
        [
            int(len(triplet_data) * 0.8),
            len(triplet_data) - int(len(triplet_data) * 0.8),
        ],
    )
    loader = torch.utils.data.DataLoader(
        dataset=train, batch_size=batch_size, shuffle=True
    )
    # test_loader = torch.utils.data.DataLoader(dataset = test_cases, batch_size = batch_size, shuffle = True)

    input_size = len(vocab.predicates) + (
        (len(vocab.variables) + len(vocab.constants)) * vocab.maxArity
    )
    model = NeuralNet(input_size, hidden_size1,
                      hidden_size2, args.embed_size).to(device)
    criterion = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - torch.cosine_similarity(x, y),
        margin=margin,
        reduction="mean",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    loss_list = []
    epoch_list = [i + 1 for i in range(num_epochs)]
    for epoch in range(num_epochs):
        running_loss = []
        for i, (anchor, positive, negative) in enumerate(loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            a_out = model(anchor)
            p_out = model(positive)
            n_out = model(negative)

            loss = criterion(a_out, p_out, n_out)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        print(np.mean(running_loss))
        loss_list.append(np.mean(running_loss))

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
    plt.plot(epoch_list, loss_list, color="red")
    plt.title("Training Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.grid(True)
    plt.show()
    if args.load_model is not None:
        model.load_state_dict(
            torch.load(args.load_model, map_location=torch.device(device))
        )
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=False)
        pos_distances = []
        neg_distances = []
        # atom_ex = None
        # atom_name = None
        # atom_ex_dist = []
        # atom_names = []
        f = True
        for i, (anchor, positive, negative) in enumerate(test_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            a_out = model(anchor).cpu().numpy().flatten()
            # if f:
            #     atom_ex = a_out
            #     atom_name = anchor.cpu().numpy().flatten()
            #     f = False
            # else:

            #     if(len(a_out)!= len(atom_ex)):
            #         continue
            #     elif(np.array_equal(atom_name, anchor.cpu().numpy().flatten())):
            #         print('here')
            #         continue
            #     elif(in_list(anchor.cpu().numpy().flatten(), atom_names)):
            #         continue
            #     x = np.dot(a_out,atom_ex)/(np.linalg.norm(a_out)*np.linalg.norm(atom_ex))
            #     atom_ex_dist.append(x)
            #     atom_names.append(anchor.cpu().numpy().flatten())

            p_out = model(positive).cpu().numpy().flatten()
            n_out = model(negative).cpu().numpy().flatten()
            pos_distances.append(
                np.dot(a_out, p_out) /
                (np.linalg.norm(a_out) * np.linalg.norm(p_out))
            )
            neg_distances.append(
                np.dot(a_out, n_out) /
                (np.linalg.norm(a_out) * np.linalg.norm(n_out))
            )
        # print(f"{atom_name}")
        # for index, val in np.ndenumerate(atom_ex_dist):
        #     print(f"{val} : {index}")

        # indicies = np.argpartition(np.array(atom_ex_dist), -6)[-5:]
        # for i in indicies:
        #     print(f"score: {atom_ex_dist[i]} name = {atom_names[i]}")
        #     name = []
        #     for j in range(len(atom_names[i])):
        #         if atom_names[i][j] == 1:
        #             name.append(j)
        #     print(name)

    print("here?")

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    axs[0].hist(pos_distances, bins=50, range=(-1, 1))
    axs[0].set_title("Unifying pair cosine similarity")
    axs[1].hist(neg_distances, bins=50, range=(-1, 1))
    axs[1].set_title("Non-unifying pair cosine similarity")
    axs[0].set_xticks([i / 2 for i in range(-2, 3)])
    axs[1].set_xticks([i / 2 for i in range(-2, 3)])
    fig.text(0.5, 0.03, "Cosine Similarity", ha="center", va="center")
    fig.text(0.03, 0.5, "Frequency", ha="center",
             va="center", rotation="vertical")
    plt.rcParams.update({"font.size": 30})
    plt.show()
    pos_distances = np.asarray(pos_distances)
    neg_distances = np.asarray(neg_distances)
    print(f"Positive mean = {np.mean(pos_distances)}")
    print(f"Negative mean = {np.mean(neg_distances)}")
    print(f"Positive standard deviation = {np.std(pos_distances)}")
    print(f"Negative standard deviation = {np.std(neg_distances)}")
    print(f"Positive median = {np.median(pos_distances)}")
    print(f"Negative median = {np.median(neg_distances)}")
    print(f"Positive max = {np.max(pos_distances)}")
    print(f"Negative max = {np.max(neg_distances)}")
    print(f"Positive min = {np.min(pos_distances)}")
    print(f"Negative min = {np.min(neg_distances)}")
    print(
        f"ks results = {scipy.stats.ks_2samp(neg_distances, pos_distances, alternative='two-sided', mode='asymp', )}"
    )
