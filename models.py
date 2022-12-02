"""Classes containing model components"""

import torch
import transformers
from torch import cuda
import torch_scatter
from transformers.trainer_utils import set_seed
import json
#file location prefixes
location_frozen_prefix = '/nlp/scr/ruthanna/cs399-f/cs399-f/model-frozen-'
location_unfrozen_prefix = '/nlp/scr/ruthanna/cs399-f/cs399-f/model-unfrozen-'
device = 'cuda:0' if cuda.is_available() else 'cpu'



class MLP(torch.nn.Module):
    '''A two-layer feedforward NN that maps the concatenated premise and hypothesis
    to output probabilities for the three NLI classes. ReLU is the non-linearity used
    '''

    def __init__(self, input_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 3)
        )
  
    def forward(self, x):
        return self.layers(x)
  
class XLMClass(transformers.XLMRobertaModel):
   '''A class for finetuning the pretrained mBERT model while keeping pretrained layers 
   frozen
   '''

   def __init__(self, config=None, pretrained_model='xlm-roberta-base', dropout=0.3, folder=None, suffix=None, seed=None, layers=None, embedding=False, experimental=False):
       if seed is not None:
           set_seed(seed)
           print('Seed F set')
       super(transformers.XLMRobertaModel, self).__init__(config)
       self.pretrained_model = pretrained_model 
       print('Pretrained model ', pretrained_model)
       self.l1 = transformers.XLMRobertaModel.from_pretrained(self.pretrained_model)
       if experimental:
           for name, param in self.l1.named_parameters():
               if name == 'embeddings.word_embeddings.weight':
                   param.requires_grad = False
       self.dropout = dropout
       self.l2 = torch.nn.Dropout(self.dropout)
       self.l3 = MLP(1536)
       print('Frozen model initialized')
   def forward(self, ids, attention_mask, token_type_ids):
       last_hidden, cls = self.l1(ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
       token_type_ids_mod = torch.clone(token_type_ids).detach()
       lengths = []
       for i in range(len(token_type_ids)):
         lengths = (ids[i] == 2).nonzero(as_tuple=True)
         for k in range(lengths[0][1] + 1, lengths[0][2]+1):
            token_type_ids_mod[i][k] = 1 #shape: batch size, num tokens
         for j in range(lengths[0][2] + 1, len(ids[i])):
            token_type_ids_mod[i][j] = 2 #shape: batch size, num tokens

       separated = torch_scatter.scatter(last_hidden, token_type_ids_mod, 1, reduce="mean")
       separated = torch.transpose(separated, 0, 1)
       premise_avg = separated[0]
       hypothesis_avg = separated[1]
       output_1 = torch.cat((premise_avg, hypothesis_avg), dim=1)
       output_1 = output_1.to(device)
       output_2 = self.l2(output_1)
       output = self.l3(output_2)
       return output   
   def toggle_freeze(self, val, all_layers=None, layers = None, string=None):
       print('Toggle freeze function ', val)
       if string is not None:
           for name, param in self.l1.named_parameters():
               if string in name:
                   param.requires_grad = val

       if layers is not None:
           for name, param in self.l1.named_parameters():
               for layer in layers:
                   if "layer.{layer}".format(layer=layer) in name:
                        param.requires_grad = val
       if all_layers is not None:
           for param in self.l1.parameters():
               param.requires_grad = val


   def print_params(self):
      print(self.pretrained_model)
      for name, param in self.l1.named_parameters():
          print(name, param.requires_grad)

class RobertaClass(transformers.RobertaModel):
   '''A class for finetuning the pretrained mBERT model while keeping pretrained layers 
   frozen
   '''

   def __init__(self, config=None, pretrained_model='roberta-base', dropout=0.3, folder=None, suffix=None, seed=None, layers=None, embedding=False, experimental=False):
       if seed is not None:
           set_seed(seed)
           print('Seed F set')
       super(transformers.RobertaModel, self).__init__(config)
       self.pretrained_model = pretrained_model 
       print('Pretrained model ', pretrained_model)
       self.l1 = transformers.RobertaModel.from_pretrained(self.pretrained_model)
       if experimental:
           for name, param in self.l1.named_parameters():
               if name == 'embeddings.word_embeddings.weight':
                   param.requires_grad = False
       self.dropout = dropout
       self.l2 = torch.nn.Dropout(self.dropout)
       self.l3 = MLP(1536)
       print('Frozen model initialized')

   def forward(self, ids, attention_mask, token_type_ids):
       last_hidden, cls = self.l1(ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
       token_type_ids_mod = torch.clone(token_type_ids).detach()
       lengths = []
       for i in range(len(token_type_ids)):
         lengths = (ids[i] == 2).nonzero(as_tuple=True)
         for k in range(lengths[0][1] + 1, lengths[0][2]+1):
            token_type_ids_mod[i][k] = 1 #shape: batch size, num tokens
         for j in range(lengths[0][2] + 1, len(ids[i])):
            token_type_ids_mod[i][j] = 2 #shape: batch size, num tokens

       separated = torch_scatter.scatter(last_hidden, token_type_ids_mod, 1, reduce="mean")
       separated = torch.transpose(separated, 0, 1)
       premise_avg = separated[0]
       hypothesis_avg = separated[1]
       output_1 = torch.cat((premise_avg, hypothesis_avg), dim=1)
       output_1 = output_1.to(device)
       output_2 = self.l2(output_1)
       output = self.l3(output_2)
       return output

   def toggle_freeze(self, val, all_layers=None, layers = None, string=None):
       print('Toggle freeze function ', val)
       if string is not None:
           for name, param in self.l1.named_parameters():
               if string in name:
                   param.requires_grad = val

       if layers is not None:
           for name, param in self.l1.named_parameters():
               for layer in layers:
                   if "layer.{layer}".format(layer=layer) in name:
                        param.requires_grad = val
       if all_layers is not None:
           for param in self.l1.parameters():
               param.requires_grad = val


   def print_params(self):
      print(self.pretrained_model)
      for name, param in self.l1.named_parameters():
          print(name, param.requires_grad)
  
class BERTClassFrozen(transformers.BertModel):
    '''A class for finetuning the pretrained mBERT model while keeping pretrained layers 
    frozen
    '''

    def __init__(self, config=None, pretrained_model='bert-base-multilingual-cased', dropout=0.3, folder=None, suffix=None, seed=None, layers=None, embedding=False, experimental=False):
        if seed is not None:
            set_seed(seed)
            print('Seed F set')
        super(transformers.BertModel, self).__init__(config)
        self.pretrained_model = pretrained_model 
        print('Pretrained model ', pretrained_model)
        self.l1 = transformers.BertModel.from_pretrained(self.pretrained_model)
        if experimental:
            for name, param in self.l1.named_parameters():
                if name == 'embeddings.word_embeddings.weight':
                    param.requires_grad = False
        self.dropout = dropout
        self.l2 = torch.nn.Dropout(self.dropout)
        self.l3 = MLP(1536)#1536
        print('hyp only')
        print('Frozen model initialized')
    def forward(self, ids, attention_mask, token_type_ids):
        last_hidden, cls = self.l1(ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
        token_type_ids_mod = torch.clone(token_type_ids).detach()
        #print('token type ids: ', token_type_ids_mod)
        lengths = []
        for i in range(len(token_type_ids)):
          length = (token_type_ids[i] == 1).nonzero(as_tuple=True)[0][-1]
          for j in range(length + 1, len(token_type_ids[i])):
             token_type_ids_mod[i][j] = 2#set all pad to 2
        separated = torch_scatter.scatter(last_hidden, token_type_ids_mod, 1, reduce="mean")
        separated = torch.transpose(separated, 0, 1)# because swapped in encoding
        premise_avg = separated[0]
        hypothesis_avg = separated[1]
        output_1 = torch.cat((premise_avg, hypothesis_avg), dim=1)
        output_1 = output_1.to(device)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
 
    def toggle_freeze(self, val, all_layers=None, layers = None, string=None):
        print('Toggle freeze function ', val)
        if string is not None:
            for name, param in self.l1.named_parameters():
                if string in name:
                    param.requires_grad = val

        if layers is not None:
            for name, param in self.l1.named_parameters():
                for layer in layers:
                    if "layer.{layer}".format(layer=layer) in name:
                         param.requires_grad = val
        if all_layers is not None:
            for param in self.l1.parameters():
                param.requires_grad = val

    def toggle_bias(self):
        print('Toggle bias function ')
        for param in self.l1.parameters():
                param.requires_grad = False

        for name, param in self.l1.named_parameters():
          if 'bias' in name:
            
            param.requires_grad = True

    def print_params(self):
       print(self.pretrained_model)
       for name, param in self.l1.named_parameters():
           print(name, param.requires_grad)
  
