from abc import ABC, abstractmethod

import basictypes
import nnreasoner
import nnunifier
from vocab import Vocabulary
from basictypes import Atom
from knowledgebase import Rule
from kbparser import parse_atom, parse_rule
import kbparser
import termwalk
import chainbased
import torch
import numpy as np

class EmbedModel(ABC):
    """
    An abstract class to represent an embedding approach for Datalog statements. Will
    return embeddings that can be used by Pytorch neural nets.
    """

    vocab: Vocabulary
    embed_size: int

    def __init__(self, vocab: Vocabulary, embed_size=20):
        self.vocab = vocab
        self.embed_size = embed_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def get_atom_embed(self, atom) -> torch.Tensor:
        pass

    @abstractmethod
    def get_rule_embed(self, rule) -> torch.Tensor:
        pass

    def get_goal_rule_embed(self, goal, rule) -> torch.Tensor:
        """
        For scoring, the input embedding will always be the embedding of the goal followed by the embedding of the rule.
        :param goal:
        :param rule:
        :return:
        """
        return torch.cat([self.get_atom_embed(goal), self.get_rule_embed(rule)]).to(self.device)


class LearnedEmbedModel(EmbedModel, ABC):
    """
    A class for an embedding model that is learned. For now, we assume that the training occurs outside of
    the class and the weights of the final NNet model are saved to a file. This class can load in the network
    from the path, and use it to embed atoms and rules.
    """
    model_path: str
    embed_net: nnunifier.NeuralNet = None

    def __init__(self, vocab: Vocabulary, embed_size: int, model_path: str, timestamp = None):
        """
        Creates a new learned, embedding model. After the model has been trained and saved, the class should
        be initialized with the path and timestamp. load() must be explicitly called before the model can
        be used to produce embeddings.
        :param vocab:
        :param embed_size:
        :param model_path:
        """

        super().__init__(vocab,embed_size)
        self.model_path = model_path

        if self.device == "cuda":
            print("Learned Embedding: Network using cuda")

    def load(self):
        # TODO: test file location and timestamp before loading
        self.embed_net.load_state_dict(torch.load(
            self.model_path, map_location=torch.device(self.device)))
        self.embed_net.eval()      # Turn off training for the network


class UnifierEmbed(LearnedEmbedModel):

    def __init__(self, vocab: Vocabulary, embed_size: int, model_path: str):

        super().__init__(vocab, embed_size, model_path)

        self.embed_net = nnunifier.NeuralNet(
            vocab.get_input_encode_size(),
            nnunifier.hidden_size1,
            nnunifier.hidden_size2,
            embed_size,
        ).to(self.device)

    def get_atom_embed(self, atom) -> torch.Tensor:
        # one_hot_atom = torch.FloatTensor(self.vocab.oneHotEncoding(self.vocab.sanitize_atom(atom))).to(self.device)
        # Spencer: Changed one hot encoding to encode atom
        one_hot_atom = self.vocab.encodeAtom(self.vocab.sanitize_atom(atom), device=self.device)
        with torch.no_grad():
            embedding = self.embed_net(one_hot_atom).to(self.device)
        return embedding

    def get_rule_embed(self, rule) -> torch.Tensor:
        """"
        For unification, the embedding of a rule is the concatenation of the head embedding with the body
        embedding. The head embedding is simply the embedding of the head atom. The body embedding is the
        sum of the embeddings of each atom in the body.
        """
        clean_rule = self.vocab.sanitize_rule(rule)
        # head_one_hot = torch.FloatTensor(self.vocab.oneHotEncoding(clean_rule.head)).to(self.device)
        head_one_hot = self.vocab.encodeAtom(clean_rule.head, device=self.device) #Spencer: encodeAtom
        with torch.no_grad():
            head_embed = self.embed_net(head_one_hot).to(self.device)
            body_embed = torch.zeros(self.embed_size, device=self.device)

            for arg in clean_rule.body:
                arg_one_hot = self.vocab.encodeAtom(arg, device=self.device) #Spencer: encodeAtom
                arg_embed = self.embed_net(arg_one_hot).to(self.device)
                body_embed += arg_embed
        embedding = torch.cat([head_embed, body_embed])
        return embedding

        # Keeping this here as a reminder of orifinal code from mr_back_reasoner.py
        # with torch.no_grad():
        #     query = {query: rule}
        #     query = kbencoder.one_hot_encode_query(query, self.vocab)
        #
        #     # Extract tensors outside the loop
        #     rule_head = self.embed_model(torch.FloatTensor(
        #         query[1]).to(device)).cpu().detach().numpy()
        #     args = torch.zeros(self.embed_size).to(device)
        #
        #     for arg in query[2]:
        #         arg = self.embed_model(torch.FloatTensor(arg).to(device))
        #         args += arg
        #
        #     args = args.cpu().detach().numpy()
        #
        #     # Convert NumPy arrays to PyTorch tensors
        #     atom = torch.FloatTensor(atom)
        #     rule_head = torch.FloatTensor(rule_head)
        #     args = torch.FloatTensor(args)
        #
        #     # Concatenate tensors using torch.cat
        #     embedding = torch.cat([atom, rule_head, args])
        # return  torch.from_numpy(embedding)


class TermWalkEmbed(EmbedModel):

    def __init__(self, vocab: Vocabulary, embed_size: int = None):
        self.update_embed_size()
        super().__init__(vocab, embed_size)

    def update_embed_size(self):
        # for now assume that behavior = 0 (first of three symbols only has 3 options)
        # TODO: double-check that this is matches what is actually constructed
        num_symbols = len(vocab.predicates) + len(vocab.constants) + len(vocab.variables) + 3
        self.embed_size = 3 + 2 * num_symbols

    def get_atom_embed(self, atom) -> torch.Tensor:
        embedding = termwalk.termwalk_representation(self.vocab.sanitize_atom(atom), self.vocab)
        return torch.from_numpy(embedding).to(self.device)

    def get_rule_embed(self, rule) -> torch.Tensor:
        embedding = termwalk.termwalk_representation(self.vocab.sanitize_rule(rule), self.vocab)
        return torch.from_numpy(embedding).to(self.device)


class ChainBasedEmbed(EmbedModel):

    def __init__(self, vocab: Vocabulary, embed_size: int):
        super().__init__(vocab, embed_size)

    def get_atom_embed(self, atom: Atom) -> torch.Tensor:
        """
        Return the chain-based embedding of an atom. It does not need to be sanitized, as all variables
        are replaced by '*'
        :param atom:
        :return:
        """
        embedding = chainbased.represent_pattern(atom, self.embed_size)
        return torch.from_numpy(embedding).to(self.device)

    def get_rule_embed(self, rule: Rule) -> torch.Tensor:
        """
        Return the chain-based embedding of a rule. It does not need to be sanitized, as all variables
        are replaced by '*'
        :param rule:
        :return:
        """
        embedding = chainbased.represent_pattern(rule, self.embed_size)
        return torch.from_numpy(embedding).to(self.device)

    # def score_rule_query_chainbased(query, rule, guidance_model):
    #     with torch.no_grad():
    #         query_vec = chainbased.represent_pattern(query, 20)
    #         rule_vec = chainbased.represent_pattern(rule, 20)
    #         embedding = np.concatenate([query_vec, rule_vec])
    #         score = nnreasoner.get_score(embedding, guidance_model)
    #         return score


# This is to test scoring before incorporating into mr_back_reasoner
def score_rule_query(embed_model, guidance_model, query, rule) -> float:
    """Evaluates query and rule and returns a score.

    :param query: A subgoal (an atom) to evaluate
    :param rule: A rule that could be used to prove the goal
    :return: A score (>=0, <=1) of the likelihood that the rule will eventually lead to a proof
    """

    embedding = embed_model.get_goal_rule_embed(query, rule)
    print("Embed size" + embedding.size())
    # changed get_score() to accept torch tensors (it was converting  numpy to torch anyway)
    # score = nnreasoner.get_score(embedding.numpy(), self.guidance_model)
    score = nnreasoner.get_score(embedding, guidance_model)
    return score

if __name__ == "__main__":

    embed_size = 50
    vocab_file = "vocab"
    # kb_file = "gameofthrones.txt"
    # kb_file = "lubm-bin-benchq.txt"
    kb_file = "randomKB.txt"
    # qfile = "test_queries.txt"
    # qfile = "debug_queries.txt"
    embed_model_path = "rKB_model.pth"
    # guidance_model_path = "uni_mr_model.pt"
    guidance_model_path = "cb_mr_model.pt"

    vocab = Vocabulary()
    vocab.init_from_vocab(vocab_file)
    kb = kbparser.parse_KB_file(kb_file)

    uni_embedding = UnifierEmbed(vocab, embed_size, embed_model_path)
    uni_embedding.load()

    cb_embedding = ChainBasedEmbed(vocab, embed_size)

    tw_embedding = TermWalkEmbed(vocab)

    cos = torch.nn.CosineSimilarity(dim=0)

    # a1: Atom = parse_atom("parent(tywin_lannister, cersei_lannister)")
    # a2: Atom = parse_atom("parent(tywin_lannister, jaime_lannister)")
    # # a3: Atom = parse_atom("parent(tywin_lannister, tyrion_lannister)")
    # a3: Atom = parse_atom(("male(gendry)"))
    #
    # e1 = cb_embedding.get_atom_embed(a1)
    # e2 = cb_embedding.get_atom_embed(a2)
    # e3 = cb_embedding.get_atom_embed(a3)
    #
    # print("Chain: " + str(a1) + "\t" + str(e1))
    # print("Chain: " + str(a2) + "\t" + str(e2))
    # print("Chain: " + str(a3) + "\t" + str(e3))
    # print("Chain sim b/t a1 and a2: " + str(cos(e1,e2)))
    # print("Chain sim b/t a1 and a3: " + str(cos(e1,e3)))
    # print()
    #
    # e1 = tw_embedding.get_atom_embed(a1)
    # e2 = tw_embedding.get_atom_embed(a2)
    # e3 = tw_embedding.get_atom_embed(a3)
    #
    # print("Termwalk: " + str(a1) + "\t" + str(e1))
    # print("Termwalk: " + str(a2) + "\t" + str(e2))
    # print("Termwalk: " + str(a3) + "\t" + str(e2))
    # print("Termwalk sim b/t a1 and a2: " + str(cos(e1,e2)))
    # print("Termwalk sim b/t a1 and a3: " + str(cos(e1,e3)))
    # print()
    # # print("One hot = \t" + str(vocab.oneHotEncoding(atom)))
    # # print()
    # # print("Unity =\t" + str(uni_embedding.get_atom_embed(atom)))
    #
    # e1 = uni_embedding.get_atom_embed(a1)
    # e2 = uni_embedding.get_atom_embed(a2)
    # e3 = uni_embedding.get_atom_embed(a3)
    #
    # print("Unify: " + str(a1) + "\t" + str(e1))
    # print("Unify: " + str(a2) + "\t" + str(e2))
    # print("Unify: " + str(a3) + "\t" + str(e2))
    # print("Unify sim b/t a1 and a2: " + str(cos(e1,e2)))
    # print("Unify sim b/t a1 and a3: " + str(cos(e1,e3)))
    # print()

    g = parse_atom("ancestor(lewyn_martell,X)")
    r1 = parse_rule("ancestor(X1,Y1) :- parent(X1,Y1)")
    r2 = parse_rule("ancestor(X1,Y1) :- parent(X1,Z1),ancestor(Z1,Y1)")

    guidance_model = nnreasoner.NeuralNet(
        nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
    ).to(uni_embedding.device)
    guidance_model.load_state_dict(
        torch.load(guidance_model_path, map_location=torch.device(
                uni_embedding.device)
        )
    )

    # print("Score (g,r1): " + str(score_rule_query(uni_embedding, guidance_model, g, r1)))
    # print("Score (g,r2): " + str(score_rule_query(uni_embedding, guidance_model, g, r2)))

    print("Model layer 1: " + str(guidance_model.l1.in_features) + ", " + str(guidance_model.l1.out_features))
    print("Score (g,r1): " + str(score_rule_query(cb_embedding, guidance_model, g, r1)))
    print("Score (g,r2): " + str(score_rule_query(cb_embedding, guidance_model, g, r2)))
