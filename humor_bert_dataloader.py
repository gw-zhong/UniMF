#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# In[2]:


'''
you can assign the maximum number number of sentences in context and what will be the maximum number of words of any sentence.

It will do left padding . It will concatenate the word embedding + covarep features + openface features

example:

if max_sen_len = 20 then the punchline sentence dimension = 20 * 752. 
    where 752 = word embedding (300) + covarep (81) + openface(371)  

if max_sen_len = 20 and max_context_len = 5 that means context can have maximum 5 sentences 
and each sentence will have maximum 20 words. The context dimension will be 5 * 20 * 752 

We will do left padding with zeros to maintaing the same dimension.

In our experiments we set max_sen_len = 20 & max_context_len = 5 
'''


class HumorBertDataset(Dataset):

    def __init__(self, mode, path, max_context_len=5, max_sen_len=20, online=False):
        # Set online to True if you need to make the dataset at the max_context_len and max_sen_len you specify.
        # Note that this will process the data in the __getitem__ method, which will cause a dramatic drop in speed.
        self.online = online
        if online:
            data_folds_file = path + "data_folds.pkl"
            openface_file = path + "openface_features_sdk.pkl"
            covarep_file = path + "covarep_features_sdk.pkl"
            language_file = path + "language_sdk.pkl"  # word_embedding_indexes_sdk
            humor_label_file = path + "humor_label_sdk.pkl"

            data_folds = load_pickle(data_folds_file)
            self.word_aligned_openface_sdk = load_pickle(openface_file)
            self.word_aligned_covarep_sdk = load_pickle(covarep_file)
            self.language_sdk = load_pickle(language_file)
            self.humor_label_sdk = load_pickle(humor_label_file)
            self.of_d = 371
            self.cvp_d = 81
            self.max_sen_len = max_sen_len
            self.id_list = data_folds[mode]

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            # If you already have a dataset with extracted features
            dataset_path = path + '[onlypunch][bert]urfunny-40.pkl'
            dataset = pickle.load(open(dataset_path, 'rb'))
            self.text = torch.tensor(dataset[mode]['text'].astype(np.float32)).cpu().detach()
            self.audio = torch.tensor(dataset[mode]['audio'].astype(np.float32)).cpu().detach()
            self.video = torch.tensor(dataset[mode]['video'].astype(np.float32)).cpu().detach()
            self.y = torch.tensor(dataset[mode]['label'].astype(np.float32)).cpu().detach()

    # right padding with zero  vector upto maximum number of words in a sentence * glove embedding dimension
    def paded_word_idx(self, seq, max_sen_len=20, left_pad=1):
        tokenized = self.tokenizer.encode_plus(
            seq, max_length=max_sen_len, add_special_tokens=True, padding='max_length', truncation=True)
        input_ids = np.array(tokenized['input_ids']).reshape(1, -1)
        attention_mask = np.array(tokenized['attention_mask']).reshape(1, -1)
        token_type_ids = np.array(tokenized['token_type_ids']).reshape(1, -1)
        pad_w = np.concatenate((input_ids, attention_mask, token_type_ids), axis=0)
        return pad_w  # (3, max_sen_len) - 3: input_ids/attention_mask/token_type_ids

    # right padding with zero  vector upto maximum number of words in a sentence * covarep dimension
    def padded_covarep_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((seq, np.zeros((max_sen_len - len(seq), self.cvp_d))), axis=0)  # (max_sen_len, 81)

    # right padding with zero  vector upto maximum number of words in a sentence * openface dimension
    def padded_openface_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((seq, np.zeros(((max_sen_len - len(seq)), self.of_d))), axis=0)  # (max_sen_len, 371)

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_cvp, max_sen_len=20, left_pad=1):

        p_seq_w = torch.FloatTensor(self.paded_word_idx(punchline_w, max_sen_len))  # (3, max_sen_len)
        p_seq_cvp = torch.FloatTensor(self.padded_covarep_features(punchline_cvp, max_sen_len))  # (max_sen_len, 81)
        p_seq_of = torch.FloatTensor(self.padded_openface_features(punchline_of, max_sen_len))  # (max_sen_len, 371)
        return p_seq_w, p_seq_cvp, p_seq_of

    def __len__(self):
        if self.online:
            return len(self.id_list)
        else:
            return len(self.y)

    def __getitem__(self, index):
        if self.online:
            hid = self.id_list[index]
            punchline_w = self.language_sdk[hid]['punchline_sentence']
            punchline_of = np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
            punchline_cvp = np.array(self.word_aligned_covarep_sdk[hid]['punchline_features'])

            # punchline feature
            x_p = self.padded_punchline_features(punchline_w, punchline_of, punchline_cvp, self.max_sen_len)

            y = torch.FloatTensor([self.humor_label_sdk[hid]])
        else:
            x_p = (self.text[index], self.audio[index], self.video[index])
            y = self.y[index]
        return x_p, y
