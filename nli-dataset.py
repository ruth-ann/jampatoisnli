"""Classes for processing and handling of the NLI datasets"""

import torch
from transformers.trainer_utils import set_seed


class NLIDataset(torch.utils.data.Dataset):
    '''A class that converts input dataframes to ids, mask, token type
    and target tensors needed for training mBERT'''

    def __init__(self, dataframe, tokenizer, max_len, seed=None):
        self.tokenizer = tokenizer
        if seed is not None:
            set_seed(seed)
            print('NLI seed set')
        self.data = dataframe
        self.premise = dataframe.premise
        self.hypothesis = dataframe.hypothesis

        self.targets = self.data.label
        self.max_len = max_len


    def __len__(self):
        return len(self.hypothesis)

    def __getitem__(self, index):
        premise = str(self.premise[index])
        hypothesis = str(self.hypothesis[index])
        inputs = self.tokenizer(
            hypothesis,
            premise,
            padding='longest', #this will actually be receiving things one by one
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        sentence = self.tokenizer.decode(ids)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
             
       }

