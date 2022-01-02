import torch
import torch.nn as nn
import torch.nn.functional as F
from data_load import DebatesDataset, UNKNOWN_TOKEN
from torch.utils.data.dataloader import DataLoader
import pickle
from sentence_transformers import SentenceTransformer


class BasicModel(nn.Module):
    """
    basic model
    BiLSTM and softmax scorer
    forward imput: sentence embeddings
    forward output: score for each class
    """
    def __init__(self,
                 embedding_dim,
                 tag_vocab_size,
                 lstm_hidden_dimension,
                 lstm_dropout=0.2,
                 lstm_n_layers=3,
                 dropout_alpha=0.2,
                 unknown_token=UNKNOWN_TOKEN,
                 embedding_model_name='paraphrase-MiniLM-L6-v2'):
        """
        :param embedding_dim: dimension of sentence embedding
        :param tag_vocab_size: number of classes
        :param lstm_hidden_dimension: dimension of lstm hidden states
        :param lstm_dropout: dropout parameter of lstm for training
        :param lstm_n_layers: number of layers of lstm
        :param dropout_alpha: dropout parameter of the model for training
        :param unknown_token: token used for representing an unknown sentence
        :param embedding_model_name: string name of the sentence embedding model
        """
        super(BasicModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_embedding_dim = embedding_dim
        self.dropout_alpha = dropout_alpha
        self.lstm = nn.LSTM(input_size=self.sentence_embedding_dim,
                            hidden_size=lstm_hidden_dimension,
                            num_layers=lstm_n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=lstm_dropout)
        self.hidden2tag = nn.Linear(lstm_hidden_dimension * 2, tag_vocab_size)
        self.unknown_token = unknown_token
        sentence_embedding_model = SentenceTransformer(embedding_model_name)
        self.unknown_token_embedding = sentence_embedding_model.encode(self.unknown_token, convert_to_tensor=True)

    def sentence_dropout(self, sentence_embedding_tensor, alpha=0.05):
        """
        replace some sentence embeddings with unknown embeddings randomly
        :param sentence_embedding_tensor: tensor [debate_length x embedding_dimension]
        :param alpha: probability for replacing a sentence
        :return: sentence_embedding_tensor after dropout
        """
        drop_idx = torch.rand(sentence_embedding_tensor.shape[1]) < alpha
        sentence_embedding_tensor[drop_idx.unsqueeze(0)] = self.unknown_token_embedding
        return sentence_embedding_tensor

    def forward(self, sentence_embedding_tensor):  # input_dimension = debate_length x embedding_dimension
        if self.training:
            sentence_embedding_tensor = self.sentence_dropout(sentence_embedding_tensor, self.dropout_alpha)
        lstm_out, _ = self.lstm(sentence_embedding_tensor.view(sentence_embedding_tensor.shape[1], 1,
                                                               -1))  # (debate_len x 2 * lstm_dim)
        tag_space = self.hidden2tag(lstm_out.view(sentence_embedding_tensor.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class AdvancedModel(nn.Module):
    """
    advanced model
    concatination of 2 different imput tensors, and then BiLSTM and softmax scorer
    forward imput: sentence embeddings and speaker-changed one hot tensor
    forward output: score for each (party) class
    """
    def __init__(self,
                 embedding_dim,
                 tag_vocab_size,
                 lstm_hidden_dimension,
                 lstm_dropout=0.2,
                 lstm_n_layers=2,
                 dropout_alpha=0.1,
                 speaker_changed_alpha=0.1,
                 unknown_token=UNKNOWN_TOKEN,
                 embedding_model_name='paraphrase-MiniLM-L6-v2'):
        """
        :param embedding_dim: dimension of sentence embedding
        :param tag_vocab_size: number of classes
        :param lstm_hidden_dimension: dimension of lstm hidden states
        :param lstm_dropout: dropout parameter of lstm for training
        :param lstm_n_layers: number of layers of lstm
        :param dropout_alpha: dropout parameter of the sentence embeddings for training
        :param speaker_changed_alpha: dropout parameter of the speaker-changed input tensor for training
        :param unknown_token: token used for representing an unknown sentence
        :param embedding_model_name: string name of the sentence embedding model
        """
        super(AdvancedModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_embedding_dim = embedding_dim
        self.dropout_alpha = dropout_alpha
        speaker_changed_one_hot_dim = 2
        self.lstm = nn.LSTM(input_size=self.sentence_embedding_dim + speaker_changed_one_hot_dim,
                            hidden_size=lstm_hidden_dimension,
                            num_layers=lstm_n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=lstm_dropout)
        self.hidden2tag = nn.Linear(lstm_hidden_dimension * 2, tag_vocab_size)
        self.unknown_token = unknown_token
        sentence_embedding_model = SentenceTransformer(embedding_model_name)
        self.unknown_token_embedding = sentence_embedding_model.encode(self.unknown_token, convert_to_tensor=True)
        self.speaker_changed_dropout = nn.Dropout(speaker_changed_alpha)

    def sentence_dropout(self, sentence_embedding_tensor, alpha=0.05):
        """
        replace some sentence embeddings with unknown embeddings randomly
        :param sentence_embedding_tensor: tensor [debate_length x embedding_dimension]
        :param alpha: probability for replacing a sentence
        :return: sentence_embedding_tensor after dropout
        """
        drop_idx = torch.rand(sentence_embedding_tensor.shape[1]) < alpha
        sentence_embedding_tensor[drop_idx.unsqueeze(0)] = self.unknown_token_embedding
        return sentence_embedding_tensor

    def forward(self, sentence_embedding_tensor, speaker_changed_tensor):  # input_dimension = debate_length x embedding_dimension
        if self.training:
            sentence_embedding_tensor = self.sentence_dropout(sentence_embedding_tensor, self.dropout_alpha)
            speaker_changed_tensor = self.speaker_changed_dropout(speaker_changed_tensor.float())
        embeds = torch.cat([sentence_embedding_tensor, speaker_changed_tensor.float()], dim=-1)
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # (debate_len x 2 * lstm_dim)
        tag_space = self.hidden2tag(lstm_out.view(sentence_embedding_tensor.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


if __name__ == '__main__':

    with open('train_advanced.pkl', 'rb') as f:
        train_advanced = pickle.load(f)
    with open('test_advanced.pkl', 'rb') as f:
        test_advanced = pickle.load(f)
    train_advanced_dataloader = DataLoader(train_advanced, shuffle=True)
    test_advanced_dataloader = DataLoader(test_advanced, shuffle=False)
    unknown_token = train_advanced.unknown_token
    model = AdvancedModel(
        embedding_dim=train_advanced.sentence_vector_dim,
        tag_vocab_size=len(train_advanced.tag_idx_mappings),
        lstm_hidden_dimension=225,
        lstm_n_layers=2,
        dropout_alpha=0.1,
        speaker_changed_alpha=0.1,
        unknown_token=unknown_token)

    model.train()

    # chack output of the model
    for sentence_embedding, speaker_changed_one_hot_tensor, party_embed_idx, debate_len in test_advanced_dataloader:
        assert sentence_embedding.shape[-1] == model.unknown_token_embedding.shape[-1]
        output = model(sentence_embedding, speaker_changed_one_hot_tensor)
        print(torch.max(output, 1))
        classifications = torch.max(output, 1)[1].detach().cpu().numpy()
        print(party_embed_idx.to("cpu"))
        truth = party_embed_idx.to("cpu").detach().cpu().numpy()
        break
