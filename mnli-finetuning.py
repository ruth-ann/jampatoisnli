"""File containing code to read in datasets and run experiments"""
from statistics import mean
import numpy as np
import pandas as pd
import torch
from nlidataset import NLIDataset
from models import BERTClassFrozen, XLMClass, RobertaClass
from torch import cuda
from sklearn import metrics
import sys
import os
import transformers
from transformers.trainer_utils import set_seed
import json
import argparse
from datetime import date
import tokenizers 
import matplotlib.pyplot as plt

mnli_train_dataset = 'datasets/multinli_1.0/multinli_1.0_train.jsonl'
lrl_dev_dataset_prefix = 'datasets/americasnli/dev/'
lrl_train_dataset_prefix = 'datasets/americasnli/finetune/'
mnli_dev_dataset = 'datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl'
device = 'cuda' if cuda.is_available() else 'cpu'


PAD_ID = 0
NUM_LABELS = 3
PAD_INDEX = 0


def read_datasets(test_dataset_names, finetune_high_dataset_name, verification_dataset_name, finetune_low_dataset_name, pretrain_low_dataset_name):
    '''Reads the datasets for training and testing from files
    '''

    label_dict_words = {'entailment': [1,0,0], 'neutral': [0,1,0], 'contradiction': [0,0,1]}
    label_dict_numbers = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}

    #MNLI train df
    mnli_train_df_unprocessed = pd.read_json(mnli_train_dataset, lines=True)
    sentence1_arr = []
    sentence2_arr = []
    y = []
    for (sentence1, sentence2, label) in zip(mnli_train_df_unprocessed.sentence1, mnli_train_df_unprocessed.sentence2, mnli_train_df_unprocessed.gold_label):
        if label != -1:
            y.append(label_dict_words[label])
            sentence1_arr.append(sentence1)
            sentence2_arr.append(sentence2)
    mnli_train_df = pd.DataFrame({'premise': sentence1_arr, 'hypothesis': sentence2_arr, 'label': y})
    test_datasets = []
    #Test dataset
    for test_dataset_name in test_dataset_names:
      print(test_dataset_name)
      lrl_dev_dataset = lrl_dev_dataset_prefix + test_dataset_name
      lrl_test_df_unprocessed = pd.read_csv(lrl_dev_dataset, delimiter='\t')
      y = []
      for label in lrl_test_df_unprocessed['label']:
          y.append(label_dict_words[label])
      lrl_test_df_unprocessed['label'] = y
      test_dataset = lrl_test_df_unprocessed[['premise', 'hypothesis', 'label']].copy()
      test_datasets.append(test_dataset)
    #LRL finetune df
    lrl_train_df = None
    if finetune_low_dataset_name is not None:
        lrl_train_dataset = lrl_train_dataset_prefix + finetune_low_dataset_name
        lrl_train_df_unprocessed = pd.read_csv(lrl_train_dataset, delimiter='\t')
        y = []
        for label in lrl_train_df_unprocessed['label']:
            y.append(label_dict_words[label])
        lrl_train_df_unprocessed['label'] = y
        lrl_train_df = lrl_train_df_unprocessed[['premise', 'hypothesis', 'label']].copy()

    #MNLI dev df
    mnli_dev_df_unprocessed = pd.read_json(mnli_dev_dataset, lines=True)
    sentence1_arr = []
    sentence2_arr = []
    y = []
    for (sentence1, sentence2, label) in zip(mnli_dev_df_unprocessed.sentence1, mnli_dev_df_unprocessed.sentence2, mnli_dev_df_unprocessed.gold_label):
        if label != -1 and label != '-':
            y.append(label_dict_words[label])
            sentence1_arr.append(sentence1)
            sentence2_arr.append(sentence2)
    mnli_dev_df = pd.DataFrame({'premise': sentence1_arr, 'hypothesis': sentence2_arr, 'label': y})
    verification_dataset = mnli_dev_df


    print('Train: ', mnli_train_df.head)
    print('Test: ', test_dataset.head)
    print('MNLI test ', verification_dataset.head)
    return test_datasets, mnli_train_df, verification_dataset, lrl_train_df, None


 

def loss_fn(outputs, targets):
    '''Defines the loss function
    '''
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def finetune(epochs, batch_size, model, optimizer,scheduler, training_loader, stop_early, curr_id, model_name):
    '''Finetunes the frozen and unfrozen models. The early stopping 
    parameter is used for debugging
    '''
    model.train()
    iteration = 0
    iterations = []
    all_losses = []
    avg_losses = []
    avg_iter = []
    avg_accum = []
    accum_iter = batch_size
    print('accum iter: ', accum_iter)
    optimizer.zero_grad()
    for epoch in range(epochs):
        losses = []
        loss_compounded = 0
        count = 0
        for i,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            if iteration%50 == 0: 
                all_losses.append(loss.item())  
                iterations.append(iteration)
            iteration += 1
            avg_accum.append(loss.item())
            if iteration%50 == 0:
                avg_losses.append(mean(avg_accum))
                avg_accum = []
                avg_iter.append(iteration)
            loss_compounded += loss.item()
            count += 1
            loss.backward()
            if ((i + 1) % accum_iter == 0) or  (i + 1 == len(training_loader)):
               optimizer.step()
               optimizer.zero_grad()

            if i%5000==0:
                print(f'Epoch: {epoch}, Current Loss:  {loss.item()}, Step: {i}')
                print(f'Epoch: {epoch}, Average over 5000 Loss:  {loss_compounded/count}, Step: {i}')
                count = 0
                loss_compounded = 0
            if stop_early:
                if i%stop_early == 0 and i!=0:
                    break
        mean_loss =sum(losses)/len(losses)
        scheduler.step(mean_loss)
    plt.plot(avg_iter, avg_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'averages-{model_name}_{curr_id}.png') 
    
    plt.plot(iterations, all_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{model_name}_{curr_id}.png')   
def validation(model, testing_loader, combined):
    '''Tests trained models. If the combined model is being tested, the frozen 
    weight is passed to the forward function as a parameter
    '''
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = None
            if combined:
              outputs = model(ids, mask, token_type_ids, frozen_weight)
            else:
              outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def pad_to_longest(batch):
     ''' Takes in a list of BATCH_SIZE NLIDataset examples, then returns
     a batched version in a dictionary containing tensors padded to the
     length of the longest example
     '''
     max_len = 0
     for b in batch:
        if len(b['ids']) > max_len:
           max_len = len(b['ids'])

     input_ids = PAD_ID * torch.ones(len(batch), max_len) #pad id is zero
     token_type_ids = PAD_INDEX * torch.ones(len(batch), max_len) #pad index is zero
     attention_mask = torch.zeros(len(batch), max_len)
     targets = torch.ones(len(batch), NUM_LABELS)

     for i in range(len(batch)):
        b = batch[i]
        for j in range(len(b['ids'])):
          input_ids[i][j] = b['ids'][j]
          token_type_ids[i][j] = b['token_type_ids'][j]
          attention_mask[i][j] = b['mask'][j]
        for k in range(NUM_LABELS):
          targets[i][k] = b['targets'][k]
     batched = {}
     batched['ids'] = input_ids
     batched['token_type_ids'] = token_type_ids
     batched['mask'] = attention_mask
     batched['targets'] = targets
     return batched

def main(epochs, true_batch_size, batch_size, lr, dropout, seed0, seed1, seed2, stop_early, suffix,curr_id, test_dataset_name, finetune_high_dataset_name, verification_dataset_name, model_1_name, layers_1,model_2_name, layers_2, finetune_low_dataset_name, pretrain_low_dataset_name, base, model_type, folder_1, folder_2):
    ''' Main flow of the program. Datasets are read, models are loaded and trained, then evaluation is
    done 
    '''

    #Create list of test dataset files
    languages = test_dataset_name
    test_dataset_names = []
    for language in languages:
      test_dataset_names.append(language + '.tsv')

    #Setup
    if seed0 is not None:
      print('Seed 0 set')
      set_seed(seed0)
    device = 'cuda' if cuda.is_available() else 'cpu'
    tokenizer = None
    mod_class = None

    #Model and tokenizer type
    print('Log: Loading tokenizer')
    if model_type == 'bert':
      tokenizer = transformers.BertTokenizer.from_pretrained(base)
      mod_class = BERTClassFrozen

    if model_type == 'xlm':
      tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(base)
      mod_class = XLMClass

    if model_type == 'roberta':
      tokenizer = transformers.RobertaTokenizer.from_pretrained(base)
      mod_class = RobertaClass
     
    max_length = None

    #Reading datasets
    print('Log: Reading datasets')
    test_dfs, finetune_high_df, verification_df, finetune_low_df, pretrain_low_df = read_datasets(test_dataset_names, finetune_high_dataset_name, verification_dataset_name, finetune_low_dataset_name, pretrain_low_dataset_name)

    finetune_high_dataset_1 = NLIDataset(finetune_high_df, tokenizer, max_length, seed0)
    finetune_high_dataset_2 = NLIDataset(finetune_high_df, tokenizer, max_length, seed0)
    test_datasets = []
    for test_df in test_dfs:
      test_dataset = NLIDataset(test_df, tokenizer, max_length, seed0)
      test_datasets.append(test_dataset)
    verification_dataset = NLIDataset(verification_df, tokenizer, max_length, seed0)
   
    finetune_high_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': int(batch_size * 0.5),
                'shuffle': False,
                'num_workers': 0
                }
    verification_params = {'batch_size': int(batch_size * 0.5),
                'shuffle': True,
                'num_workers': 0
                }
 
    test_loaders = []
    finetune_high_loader_1 = torch.utils.data.DataLoader(finetune_high_dataset_1, collate_fn=pad_to_longest, **finetune_high_params)
    finetune_high_loader_2 = torch.utils.data.DataLoader(finetune_high_dataset_2, collate_fn=pad_to_longest, **finetune_high_params)
    
    for test_dataset in test_datasets:
      test_loader= torch.utils.data.DataLoader(test_dataset, collate_fn=pad_to_longest ,**test_params)
      test_loaders.append(test_loader)
    verification_loader = torch.utils.data.DataLoader(verification_dataset, collate_fn=pad_to_longest ,**verification_params)
 
    #Loading all models
    print('Log: Loading models\n')
    model_1_folder = 'unfrozen' + '-' + curr_id + '/'
    model_2_folder = 'frozen' + '-' + curr_id + '/'

    config = transformers.AutoConfig.from_pretrained(base)
    model_1 = mod_class(config, pretrained_model=base, dropout=dropout, folder=model_1_folder, seed=seed1, layers=layers_1)
    model_2 = mod_class(config, pretrained_model=base, dropout=dropout, folder=model_2_folder, seed=seed2, layers=layers_2)
    model_2.toggle_freeze(False, all_layers = 1)
    model_2.toggle_freeze(True, layers=layers_2)

    model_1.to(device)
    model_2.to(device)

    print('\n Model 1')
    model_1.print_params()
    print('\n Model 2')
    model_2.print_params()

    #Training
    print('Log: Training \n\n\n')
    optimizer_1 = torch.optim.AdamW(params =  model_1.parameters(), lr=lr) 
    optimizer_2 = torch.optim.AdamW(params =  model_2.parameters(), lr=lr)
    scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, patience=5, factor=0.1, verbose=True)
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, patience=5, factor=0.1, verbose=True)

    finetune(epochs, true_batch_size/batch_size, model_2, optimizer_2, scheduler_2,  finetune_high_loader_2, stop_early, str(curr_id), 'frozen')
    finetune(epochs, true_batch_size/batch_size, model_1, optimizer_1, scheduler_1, finetune_high_loader_1, stop_early, str(curr_id), 'unfrozen')   

    if not os.path.exists(model_1_folder):
        os.makedirs(model_1_folder)
    if not os.path.exists(model_2_folder):
        os.makedirs(model_2_folder)

    model_1.save_pretrained(model_1_folder, push_to_hub=False)
    model_2.save_pretrained(model_2_folder, push_to_hub=False)


    #Evaluation
    print('Log: validation\n\n\n')
    verify_model_1_outputs, verify_model_1_targets = validation(model_1, verification_loader, False)
    print('\nMNLI ', model_1_name, ' outputs')
    print(verify_model_1_outputs[:5])
    print('\n')
    verify_model_1_targets = np.argmax(verify_model_1_targets, axis=1).reshape(-1, 1)
    verify_model_1_outputs = np.argmax(verify_model_1_outputs, axis=1).reshape(-1, 1)
    verify_model_1_accuracy = metrics.accuracy_score(verify_model_1_targets, verify_model_1_outputs)
    print(f"\n\nMNLI control accuracy score = {verify_model_1_accuracy}")
    print(np.stack([verify_model_1_outputs[:50].T, verify_model_1_targets[:50].T]).tolist())

    verify_model_2_outputs, verify_model_2_targets = validation(model_2, verification_loader, False)
    print('\nMNLI ', model_2_name, ' outputs')
    print(verify_model_2_outputs[:5])
    print('\n')
    verify_model_2_targets = np.argmax(verify_model_2_targets, axis=1).reshape(-1, 1)
    verify_model_2_outputs = np.argmax(verify_model_2_outputs, axis=1).reshape(-1, 1)
    verify_model_2_accuracy = metrics.accuracy_score(verify_model_2_targets, verify_model_2_outputs)
    print(f"\n\nMNLI experimental accuracy score = {verify_model_2_accuracy}")
    print(np.stack([verify_model_2_outputs[:50].T, verify_model_2_targets[:50].T]).tolist())

if __name__=="__main__":
    print('Log: Reading command line args\n')

    argp = argparse.ArgumentParser()

    #params
    argp.add_argument('epochs', help='Epochs', type=int)
    argp.add_argument('true_batch_size', help='Batch size', type=int)
    argp.add_argument('batch_size', help='Batch size', type=int)
    argp.add_argument('lr', help='Learning rate', type=float)
    argp.add_argument('dropout', help='Dropout', type=float)
    argp.add_argument('s0', help='Random seed for main', type=int)
    argp.add_argument('s1', help='Random seed for first model', type=int)
    argp.add_argument('s2', help='Random seed for second model', type=int)

    #flags
    argp.add_argument('stop_early', help='For debugging', type=int)
    #storage
    argp.add_argument('suffix', help='Suffix for saving the model', type=str)
    argp.add_argument('id', help='ID', type=str)

    #datasets
    argp.add_argument('fhd', help='HRL dataset for finetuning the model', type=str)
    argp.add_argument('vd', help='Dataset for verifying the model', type=str)

    #models
    argp.add_argument('--m1', help='Name of first model to compose', type=str, default='unfrozen')
    argp.add_argument('--l1',  nargs='+', help='Layers to unfreeze in first model', type=str, default=None)

    argp.add_argument('--m2', help='Name of second model to compose', type=str, default='frozen')
    argp.add_argument('--l2',  nargs='+', help='Layers to unfreeze in second model', type=str, default=None)
    #optional args
    argp.add_argument('--fld', help='LRL language dataset for finetuning the model', type=str, default=None) 
    argp.add_argument('--pld', help='LRL language dataset for further pretraining the model', type=str, default=None)
    argp.add_argument('--base', help='Base model', type=str, default='bert-base-multilingual-cased')
    argp.add_argument('--model_type', help='Model type', type=str, default='bert')

    argp.add_argument('--f1', help='Base model', type=str, default=None)
    argp.add_argument('--f2', help='Base model', type=str, default=None)

    argp.add_argument('--td', nargs='+', help='Dataset for testing the model', type=str, default=None)
    args = argp.parse_args()
    print('Command line args: epochs, true_batch_size, batch_size, lr, dropout, s0, s1, s2, training, try_all_weights, stop_early, suffix, id_pre, id, td, fhd, vd, m1, l1, fl1, m2, l2, fl2, ad, fld, pld, base, emb, freeze, mnli_tar:\n')
    print(args.epochs, args.true_batch_size, args.batch_size, args.lr, args.dropout, args.s0, args.s1, args.s2,args.stop_early, args.suffix,  args.id,  args.td, args.fhd, args.vd, args.m1, args.l1, args.m2, args.l2, args.fld, args.pld, args.base, args.model_type, args.f1, args.f2)

    main(args.epochs, args.true_batch_size, args.batch_size, args.lr, args.dropout, args.s0, args.s1, args.s2,args.stop_early, args.suffix,  args.id,  args.td, args.fhd, args.vd, args.m1, args.l1, args.m2, args.l2, args.fld, args.pld, args.base, args.model_type, args.f1, args.f2)

