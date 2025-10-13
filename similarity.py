from vocab import Vocabulary
from kbparser import parse_atom
from prints import print_progress_bar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nnunifier
import scipy
import torch
import csv
from atomgenerator import generate_single_atom, new_triplets, encode_triplets

# vocab.py and triplets.csv requried
# triplets.csv only required if EXTRACT is set to True

EXTRACT = True
PLOT_SAMPLES = 20000 # Number of unique samples to generate for plotting

def unique_atoms(anchors, size):
		atoms = set(anchors)
		while len(atoms) <= size + 100:
			new_atom = generate_single_atom(vocab)
			atoms.add(new_atom)
			print_progress_bar(
					len(atoms), size, length=20, suffix="Anchors generated")
		
		print()

		atoms = atoms - set(anchors)
		return list(atoms)

def extract_triplet_pairs(vocab: Vocabulary, triplet_path="triplets.csv"):
	# df = pd.read_csv(triplet_path, delimiter='\t')
	trips = set()
	real_trips = []
	anchors = []

	# for i, row in df.iterrows():
	# 	anchor_atom = parse_atom(row.iloc[0])
	# 	positive_atom = parse_atom(row.iloc[1])
	# 	negative_atom = parse_atom(row.iloc[2])

	# 	anchors.append(anchor_atom)
	# 	trips.add(frozenset({anchor_atom, positive_atom, negative_atom}))
	# 	print_progress_bar(
	# 		i+1, len(df.index), length=20, suffix="Triplet extraction")

	print()

	#Generate unique triplets from ones already given
	anchors = list(set(anchors))
	init_len = len(trips)
	while len(trips) <= init_len + PLOT_SAMPLES:
		new_trips = new_triplets(vocab, unique_atoms(anchors, len(anchors)+PLOT_SAMPLES), "similarity_triplets.csv", 1)
		for trip in new_trips:
			size = len(trips)
			if size > init_len + PLOT_SAMPLES:
				break
			trips.add(frozenset({trip[0], trip[1], trip[2]}))
			#Representative frozen set, if length changes, add new trip to the real set.
			if len(trips) != size:
				real_trips.append([trip[0], trip[1], trip[2]])
			
		print_progress_bar(
				len(real_trips), (PLOT_SAMPLES), length=20, suffix="Actual Progress")
		print()
	
	#One hot encode, return
	encoded_trips = encode_triplets(vocab, real_trips)

	return encoded_trips, real_trips

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vocab = Vocabulary()
vocab.init_from_vocab("vocab")
input_size = len(vocab.predicates) + \
    ((len(vocab.variables) + len(vocab.constants)) * vocab.maxArity)
model = nnunifier.NeuralNet(input_size,
							nnunifier.hidden_size1,
							nnunifier.hidden_size2,
							50).to(device)
model.load_state_dict(torch.load("rKB_model.pth",
									map_location=torch.device(device)))
model.eval()

if EXTRACT:
		onehot_trips, real_trips = extract_triplet_pairs(vocab)
		ancs, poss, negs = onehot_trips[0], onehot_trips[1], onehot_trips[2]
else:
		with open("train_anchors.csv", 'r') as x:
			ancs = list(csv.reader(x, delimiter=","))

		with open("train_positives.csv", 'r') as x:
			poss = list(csv.reader(x, delimiter=","))

		with open("train_negatives.csv", 'r') as x:
			negs = list(csv.reader(x, delimiter=","))


triplet_data = nnunifier.AtomData(
	ancs, poss, negs
)

test = triplet_data
test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=False)
pos_distances, neg_distances = [], []
neg_outsiders, pos_outsiders = [], []

f = True
for i, (anchor, positive, negative) in enumerate(test_loader):
	anchor = anchor.to(device)
	positive = positive.to(device)
	negative = negative.to(device)

	a_out = model(anchor)
	p_out = model(positive)
	n_out = model(negative)

	pos_similarity = torch.nn.PairwiseDistance()(a_out, p_out)
	pos_distances.append(pos_similarity)
	if (pos_similarity > 1):
		pos_outsiders.append([pos_similarity.item(), real_trips[i][0], real_trips[i][1]])

	neg_similarity = torch.nn.PairwiseDistance()(a_out, n_out)
	neg_distances.append(neg_similarity)
	if (neg_similarity < 1):
		neg_outsiders.append([neg_similarity.item(), real_trips[i][0], real_trips[i][2]])
	
	print_progress_bar(
		i+1, PLOT_SAMPLES, length=20, suffix="Setting distance pairs")

with open('pos_outsiders.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(pos_outsiders)

with open('neg_outsiders.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(neg_outsiders)

fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)

for i, elm in enumerate(pos_distances):
		pos_distances[i] = elm.detach().numpy().flatten()[0]
	
for i, elm in enumerate(neg_distances):
		neg_distances[i] = elm.detach().numpy().flatten()[0]

axs[0][0].hist(pos_distances, bins=50, range=(min(pos_distances), max(pos_distances)))
axs[0][0].set_title("Positive Pair Similarity")
axs[0][0].set_xticks(np.append(np.arange(0, max(pos_distances)+1, 3.0),[1]))

axs[0][1].hist(neg_distances, bins=50, range=(min(neg_distances), max(neg_distances)))
axs[0][1].set_title("Negative Pair Similarity")
axs[0][1].set_xticks(np.append(np.arange(0, max(neg_distances)+1, 3.0),[1]))

axs[1][0].hist(pos_distances, bins=50, range=(0, 3))
axs[1][0].set_title("Positive Pair (Focused)")
axs[1][0].set_xticks(np.arange(0, 3.1, 0.5))

axs[1][1].hist(neg_distances, bins=50, range=(0, 3))
axs[1][1].set_title("Negative Pair (Focused)")
axs[1][1].set_xticks(np.arange(0, 3.1, 0.5))

fig.text(0.5, 0.03, "Pairwise Similarity", ha="center", va="center")
fig.text(0.03, 0.5, "Frequency", ha="center",
			va="center", rotation="vertical")
plt.rcParams.update({"font.size": 30})
#plt.show()
plt.savefig('pairwise_similarity.png')

pos_distances = np.asarray(pos_distances)
neg_distances = np.asarray(neg_distances)
print()
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