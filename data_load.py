import pickle
from torch.utils.data.dataset import Dataset
import torch
import pandas
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.nn.functional import one_hot
from preprocess import read_data


def get_tag_vocabs(debates_dfs):
    """
        Extract vocabs from list of dataframes. Return basic model tag2idx (binary) and Return advanced model tag2idx (trinary).
        :param debates_dfs: a dataframe with all debates and their sentences and tags
            Return:
              - basic_tag2idx
              - advanced_tag2idx
    """
    basic_tag_dict = defaultdict(int)
    advanced_tag_dict = defaultdict(int)
    for df in debates_dfs:
        for i, row in df.iterrows():
            basic_tag, advanced_tag = row['speaker_changed'], row['party']
            basic_tag_dict[basic_tag] += 1
            advanced_tag_dict[advanced_tag] += 1
    return basic_tag_dict, advanced_tag_dict


UNKNOWN_TOKEN = '[UNK]'


class DebatesDataset(Dataset):
    """
        a dataset in accordance to the task
    """
    def __init__(self, basic_tag_dict, advanced_tag_dict, data, model_type='speaker changed only'):
        """
            :param basic_tag_dict: dictionary with speaker_changed labels as keys
            :param advanced_tag_dict: dictionary with party labels as keys,
                i.e. {'democrat': 0, 'other': 1, 'republican': 2}
            :param data: list of dataframes of all the debates. columns=['speaker','sentence','speaker_changed','party']
            :param model_type: type of the dataset
                'speaker changed only': sentence_embeds as input and speaker_changed as label
                'party only': sentence_embeds as input and party as label
                'advanced': sentence_embeds and speaker_changed as inputs and party as label
        """
        super().__init__()
        self.data = data
        if len(self.data) == 0:
            return
        self.basic_tag_idx_mappings, self.basic_idx_tag_mappings = self.init_tag_vocab(basic_tag_dict)
        self.advanced_tag_idx_mappings, self.advanced_idx_tag_mappings = self.init_tag_vocab(advanced_tag_dict)
        self.model_type = model_type
        self.tag_idx_mappings =\
            self.basic_tag_idx_mappings if self.model_type == 'speaker changed only' else self.advanced_tag_idx_mappings
        self.idx_tag_mappings =\
            self.basic_idx_tag_mappings if self.model_type == 'speaker changed only' else self.advanced_idx_tag_mappings
        self.unknown_token = UNKNOWN_TOKEN
        self.debates_lens = [len(debate) for debate in self.data]
        self.max_seq_len = max(self.debates_lens)
        self.debates_dataset = self.convert_debates_to_dataset()
        self.sentence_vector_dim = self.debates_dataset[0][0].size(-1)

    def __len__(self):
        return len(self.debates_dataset)

    def __getitem__(self, index):
        """in speaker-changed-only model/party-only models:
            sentence_embed_tansor, tag_embed_idx, debates_len = self.debates_dataset[index],
            where tag is speaker changed indicator or party label.
        in advanced model:
            sentence_embed_tensor, speaker_changed_embed_idx, party_embed_idx, debates_len = self.debates_dataset[index]
        """
        return self.debates_dataset[index]

    def init_tag_vocab(self, tag_dict):
        """
        create mapping between indices and tags.
        :param tag_dict: dictionary with tags as keys
            Return:
              - tag_idx_mappings: dictionary
              - idx_tag_mappings: list
        """
        tag_idx_mappings, idx_tag_mappings = {}, []
        for i, tag in enumerate(sorted(tag_dict.keys())):
            tag_idx_mappings[tag] = int(i)
            idx_tag_mappings.append(tag)
        return tag_idx_mappings, idx_tag_mappings

    def get_tag_vocab(self):
        return self.tag_idx_mappings, self.idx_tag_mappings

    def convert_debates_to_dataset(self):
        """
        converts list of debates dataframes to dictionary of samples
        :return: a dictionary with indices as keys and samples as values.
                 in speaker-changed-only model type, sample is a tuple (sentence_embed_tansor, speaker_changed_idx, debates_len).
                 in party-only model type, sample is a tuple (sentence_embed_tansor, party_embed_idx, debates_len).
                 in advanced model type, sample is a tuple:
                    (sentence_embed_tansor, speaker_changed_one_hot_tensor, party_embed_idx, debates_len).
        """
        debate_sentence_embedding_list = list()
        debate_speaker_changed_idx_list = list()
        debate_party_idx_list = list()
        debate_speaker_changed_one_hot_list = list()
        debate_len_list = list()
        sentence_embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        for debate_idx, debate in enumerate(self.data):
            speaker_changed_idx_list = list()
            party_idx_list = list()
            sentences_embedding_tensor =\
                sentence_embedding_model.encode(debate['sentence'].to_list(), convert_to_tensor=True)
            for i, row in debate.iterrows():
                speaker_changed_tag = row['speaker_changed']
                party_tag = row['party']
                speaker_changed_idx_list.append(self.basic_tag_idx_mappings.get(speaker_changed_tag))
                party_idx_list.append(self.advanced_tag_idx_mappings.get(party_tag))
            debate_len = len(sentences_embedding_tensor)

            speaker_changed_idx_tensor = torch.tensor(speaker_changed_idx_list, dtype=torch.long, requires_grad=False)
            debate_speaker_changed_idx_list.append(speaker_changed_idx_tensor)
            debate_party_idx_list.append(torch.tensor(party_idx_list, dtype=torch.long, requires_grad=False))
            debate_speaker_changed_one_hot_list.append(one_hot(speaker_changed_idx_tensor))
            debate_sentence_embedding_list.append(sentences_embedding_tensor)
            debate_len_list.append(debate_len)

        if self.model_type == 'party only':
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(debate_sentence_embedding_list,
                                                                     debate_party_idx_list,
                                                                     debate_len_list))}
        elif self.model_type == 'advanced':
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(debate_sentence_embedding_list,
                                                                     debate_speaker_changed_one_hot_list,
                                                                     debate_party_idx_list,
                                                                     debate_len_list))}
        elif self.model_type == 'speaker changed only':
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(debate_sentence_embedding_list,
                                                                     debate_speaker_changed_idx_list,
                                                                     debate_len_list))}
        else:
            raise Exception("invalid model type")


def create_datasets(path, model_type='speaker changed only'):
    """
    read data from files and create train and test datasets
    :param path: path to a directory which contains text transcripts of all debates
    :param model_type: model_type: type of the datasets
    :return: train dataset with 40 debates and test dataset with 6 debates
    """
    debates_dfs = read_data(path)
    # with open('debates_dfs.pkl', 'rb') as f:
    #     debates_dfs = pickle.load(f)
    data_train, data_test = train_test_split(debates_dfs, test_size=0.13, random_state=2)
    basic_tag_dict, advanced_tag_dict = get_tag_vocabs(debates_dfs)
    train = DebatesDataset(basic_tag_dict, advanced_tag_dict, data_train, model_type=model_type)
    test = DebatesDataset(basic_tag_dict, advanced_tag_dict, data_test, model_type=model_type)
    return train, test


if __name__ == '__main__':
    path = 'debates'
    train, test = create_datasets(path, 'speaker changed only')
    with open('train_speaker_changed.pkl','wb') as f:
        pickle.dump(train,f)
    with open('test_speaker_changed.pkl', 'wb') as f:
        pickle.dump(test, f)

    party_train, party_test = create_datasets(path, 'party only')
    with open('train_party.pkl','wb') as f:
        pickle.dump(party_train,f)
    with open('test_party.pkl', 'wb') as f:
        pickle.dump(party_test, f)

    advanced_train, advanced_test = create_datasets(path, 'advanced')
    with open('train_advanced.pkl','wb') as f:
        pickle.dump(advanced_train,f)
    with open('test_advanced.pkl', 'wb') as f:
        pickle.dump(advanced_test, f)

    # print proportions of classes
    for set in [train, test]:
        count_democrat = 0
        count_republican = 0
        count_other = 0
        count_speaker_changed = 0
        count_speaker_not_changed = 0
        for debate in set.data:
            for index, row in debate.iterrows():
                if row['party'] == 'democrat':
                    count_democrat += 1
                elif row['party'] == 'republican':
                    count_republican += 1
                else:
                    count_other += 1
                if row['speaker_changed']:
                    count_speaker_changed += 1
                else:
                    count_speaker_not_changed += 1
        sum = count_other + count_democrat + count_republican
        print("number of sentences is : ", sum)
        print("democrat proportion is : ", 100 * count_democrat / sum)
        print("republican proportion is : ", 100 * count_republican / sum)
        print("other proportion is : ", 100 * count_other / sum)
        print("speaker changed proportion is :", 100 * count_speaker_changed / sum)
        print("speaker changed not proportion is :", 100 * count_speaker_not_changed / sum)
