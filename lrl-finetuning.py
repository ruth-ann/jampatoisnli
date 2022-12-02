
from statistics import mean 
import numpy as np 
import pandas as pd 
import torch 
from nlidataset import NLIDataset 
from model2 import XLMClassUnfrozen,XLMClassFrozen, XLMClassCombine, BERTClassCombine, BERTClassFrozen, BERTClassUnfrozen, XLMClass, RobertaClass 
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
import copy

mnli_dev_dataset = 'datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl'
lrl_dev_dataset_prefix = 'datasets/americasnli/test/'
lrl_train_dataset_prefix = 'datasets/americasnli/finetune/'
PAD_ID = 0
NUM_LABELS = 3
PAD_INDEX = 0 
device = 'cuda' if cuda.is_available() else 'cpu'

def read_datasets(test_dataset_names, patois_test_set):

    label_dict_words = {'entailment': [1,0,0], 'neutral': [0,1,0], 'contradiction': [0,0,1]}
    label_dict_numbers = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}


    #AmericasNLI dataset
    test_datasets = []
    finetune_datasets = []
    for test_dataset_name in test_dataset_names[:-1]:
      print(test_dataset_name)
      lrl_dev_dataset = lrl_dev_dataset_prefix + test_dataset_name + '-patois-test.tsv'
      lrl_test_df_unprocessed = pd.read_csv(lrl_dev_dataset, delimiter='\t')
      y = []
      word_label = []
      for label in lrl_test_df_unprocessed['label']:
          y.append(label_dict_words[label])
          word_label.append(label)
      lrl_test_df_unprocessed['label'] = y
      lrl_test_df_unprocessed['word_label'] = word_label
      test_dataset = lrl_test_df_unprocessed[['premise', 'hypothesis', 'label', 'word_label']].copy()
      test_datasets.append(test_dataset)


      lrl_train_dataset = lrl_train_dataset_prefix + test_dataset_name + '-patois-train.tsv'
      lrl_train_df_unprocessed = pd.read_csv(lrl_train_dataset, delimiter='\t')
      y = []
      word_label = []
      for label in lrl_train_df_unprocessed['label']:
          y.append(label_dict_words[label])
          word_label.append(label)
      lrl_train_df_unprocessed['label'] = y
      lrl_train_df_unprocessed['word_label'] = word_label
      lrl_train_df = lrl_train_df_unprocessed[['premise', 'hypothesis', 'label', 'word_label']].copy()
      finetune_datasets.append(lrl_train_df)  
    #Patois dataset

    patois_test = 'datasets/patoisnli/jampatoisnli-test.csv'
    if patois_test_set != "":
      patois_test = 'datasets/patoisnli/' + patois_test_set
    patois_test_df_unprocessed = pd.read_csv(patois_test, delimiter=',')
    y = []
    word_label = []
    for label in patois_test_df_unprocessed['label']:
        y.append(label_dict_words[label])
        word_label.append(label)
    patois_test_df_unprocessed['label'] = y

    patois_test_df_unprocessed['word_label'] = word_label
    patois_test_dataset = patois_test_df_unprocessed[['premise', 'hypothesis', 'label', 'word_label']].copy()


    patois_train = 'datasets/patoisnli/jampatoisnli-train.csv'
    patois_train_df_unprocessed = pd.read_csv(patois_train, delimiter=',')
    y = []
    word_label = []
    for label in patois_train_df_unprocessed['label']:
        y.append(label_dict_words[label])

        word_label.append(label)
    patois_train_df_unprocessed['label'] = y
    
    patois_train_df_unprocessed['word_label'] = word_label
    patois_train_dataset = patois_train_df_unprocessed[['premise', 'hypothesis', 'label', 'word_label']].copy()
    #MNLI dev df
    mnli_dev_df_unprocessed = pd.read_json(mnli_dev_dataset, lines=True)
    sentence1_arr = []
    sentence2_arr = []
    y = []
    word_label = []
    for (sentence1, sentence2, label) in zip(mnli_dev_df_unprocessed.sentence1, mnli_dev_df_unprocessed.sentence2, mnli_dev_df_unprocessed.gold_label):
        if label != -1 and label != '-':
            y.append(label_dict_words[label])
            sentence1_arr.append(sentence1)
            sentence2_arr.append(sentence2)
            word_label.append(label)
    mnli_dev_df = pd.DataFrame({'premise': sentence1_arr, 'hypothesis': sentence2_arr, 'label': y, 'word_label': word_label})
    verification_dataset = mnli_dev_df
    return test_datasets, finetune_datasets, verification_dataset, patois_test_dataset, patois_train_dataset

def finetune(epochs, batch_size, model, optimizer,scheduler, training_loader, stop_early, curr_id, model_name, iter_setup):
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
    end = False
    true_iteration = 0
    print('accum iter: ', accum_iter)
    optimizer.zero_grad()
    for epoch in range(epochs):
        print(end)
        if end == True:
          print('ending early')
          break
        losses = []
        loss_compounded = 0
        count = 0
        print(len(training_loader))
        for i,data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)
            #optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())#detach, numpy- just make them into floats- detach, tonumpy

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
               #print('OPTIM STEP')
               true_iteration += 1
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
            if iter_setup != 0:
              print('true iteration, iter setup: ',  true_iteration, iter_setup, iter_setup == true_iteration)
              if int(true_iteration) == int(iter_setup):
                print('ending epoch: ', epoch, 'ending iteration: ', true_iteration)
                end = True
                break
        mean_loss =sum(losses)/len(losses)
        scheduler.step(mean_loss)
    print('iteration, true_ iteration: ', iteration, true_iteration)
    #return iterations, all_losses
    plt.plot(avg_iter, avg_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'averages-{model_name}_{curr_id}.png')

    plt.plot(iterations, all_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{model_name}_{curr_id}.png')

def pad_to_longest(batch):
     ''' Takes in a list of BATCH_SIZE NLIDataset examples, then returns
     a batched version in a dictionary containing tensors padded to the
     length of the longest example
     '''
     max_len = 0
     for b in batch:
        if len(b['ids']) > max_len:
           max_len = len(b['ids'])
     #ones tensors are used in the initializations below as a sanity check to ensure that data is being copied over 
     #correctly

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

def loss_fn(outputs, targets):
    '''Defines the loss function
    '''

    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def main(models, datasets, epochs, true_batch, batch_size, lr, dropout, base, model_type, curr_id, seed, setting, stop_early, layers, iter_setup, verify, test_set):
  if seed is not None:
      print('Seed 0 set')
      set_seed(seed)
  max_length = None
  device = 'cuda' if cuda.is_available() else 'cpu'
  tokenizer = None
  mod_class = None
  print('Log: Loading tokenizer')
  if model_type == 'bert':
    tokenizer = transformers.BertTokenizer.from_pretrained(base) 
    mod_class = BERTClassFrozen 
    unfrozen_mod_class = BERTClassUnfrozen 

  if model_type == 'xlm': 
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(base) 
    mod_class = XLMClass 
  if model_type == 'roberta': 
    tokenizer = transformers.RobertaTokenizer.from_pretrained(base) 
    mod_class = RobertaClass 
  

  test_dfs, finetune_low_dfs, verification_df, patois_test_df, patois_finetune_df = read_datasets(datasets, test_set)
  test_datasets = []
  finetune_datasets = []
  test_dfs.append(patois_test_df)
  for test_df in test_dfs:
      test_dataset = NLIDataset(test_df, tokenizer, max_length, seed)
      test_datasets.append(test_dataset)
  verification_dataset = NLIDataset(verification_df, tokenizer, max_length, seed)
  finetune_low_dfs.append(patois_finetune_df)
  for finetune_low_df in finetune_low_dfs:
      finetune_low_dataset = NLIDataset(finetune_low_df, tokenizer, max_length, seed)
      finetune_datasets.append(finetune_low_dataset)
  patois_finetune_dataset = NLIDataset(patois_finetune_df, tokenizer, max_length, seed)
  finetune_datasets.append(patois_finetune_dataset)

  test_params = {'batch_size': int(batch_size),
              'shuffle': False,
              'num_workers': 0
              }
  verification_params = {'batch_size': int(batch_size),
              'shuffle': True,
              'num_workers': 0
              }

  test_loaders = []
  finetune_loaders = []


  verification_loader = torch.utils.data.DataLoader(verification_dataset, collate_fn=pad_to_longest ,**verification_params)
  for test_dataset in test_datasets:
    test_loader= torch.utils.data.DataLoader(test_dataset, collate_fn=pad_to_longest ,**test_params)
    test_loaders.append(test_loader)
  accuracies = []
  languages = []
  all_model_names = []
  num_examples_arr = []
  model_names = None
  if not 'unfrozen' in  setting:
    model_names = ['frozen-' + model for model in models]
  else:
    model_names = ['unfrozen-' + model for model in models]
  for model in models:
    model_name = None
    if not 'unfrozen' in setting:
      model_name = 'frozen-' + model
    else:
      model_name = 'unfrozen-' + model
    model_obj =  mod_class.from_pretrained(model_name, pretrained_model=base, dropout=dropout, seed=seed, layers=layers)

    model_obj.to(device)
    #included for bit fit
    #model_obj.toggle_bias()
    #included for bit fit
    model_obj.print_params()
    if verify:
      verify_outputs, verify_targets = validation(model_obj, verification_loader, False) 
      verify_targets = np.argmax(verify_targets, axis=1).reshape(-1, 1) 
      verify_outputs = np.argmax(verify_outputs, axis=1).reshape(-1, 1) 
      verify_accuracy = metrics.accuracy_score(verify_targets, verify_outputs) 
      accuracies.append(verify_accuracy)
      languages.append('english')
      num_examples_arr.append(0)
      all_model_names.append(model_name)
      df = pd.DataFrame(list(zip(all_model_names, languages, num_examples_arr, accuracies)), columns=['Model Name', 'Language', 'Num Examples', 'Accuracy']) 
      df.to_csv(f'{curr_id}-{setting}.csv')    
    else:
      for i in range(len(datasets)):
        max_examples = len(finetune_datasets[i])
        num_examples_pow = 0
        num_examples = pow(2, num_examples_pow) 
        cont = 0
        model_copy = copy.deepcopy(model_obj)
        model_copy = model_copy.to(device)
        model_copy.toggle_freeze(True, all_layers = 1)

        #for bitfit
          
        #model_copy.toggle_bias()
        #for bitfit

        model_copy.print_params()
        model_outputs, model_targets = validation(model_copy, test_loaders[i], False)
        model_targets = np.argmax(model_targets, axis=1).reshape(-1, 1)
        model_outputs = np.argmax(model_outputs, axis=1).reshape(-1, 1)
        model_accuracy = metrics.accuracy_score(model_targets, model_outputs)
        results_df = pd.DataFrame({'targets':model_targets.squeeze(), 'outputs':model_outputs.squeeze()})
        results_df.to_csv(f'results_{curr_id}_0_{datasets[i]}.csv')

        accuracies.append(model_accuracy)
        languages.append(datasets[i])
        num_examples_arr.append(0)
        all_model_names.append(model_name)

        while cont != 2:
          print(datasets[i])
          finetune_low_df_mini =  finetune_low_dfs[i].groupby('word_label').apply(lambda x: x.sample(n=num_examples))
          finetune_low_df_mini = finetune_low_df_mini.sample(frac=1) 
          print(finetune_low_df_mini) 
          finetune_params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0
              }
          finetune_dataset = NLIDataset(finetune_low_df_mini, tokenizer, max_length, seed)
          finetune_loader= torch.utils.data.DataLoader(finetune_dataset, collate_fn=pad_to_longest ,**finetune_params)
             
          model_copy = copy.deepcopy(model_obj)
          model_copy = model_copy.to(device)
          model_copy.toggle_freeze(True, all_layers = 1)
          #for bitfit
          
          #model_copy.toggle_bias()
          #for bitfit
          model_copy.print_params()
          curr_iter_setup = iter_setup
          curr_lr = lr
          if num_examples <= 16:
            curr_iter_setup /= 2
            curr_lr /= 5
          optimizer = torch.optim.AdamW(params =  model_copy.parameters(), lr=curr_lr)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5, verbose=True)
          this_batch = None
          if iter_setup == 0 or num_examples > true_batch:
            this_batch = true_batch/batch_size
          else:
            this_batch = num_examples
          print('num examples, batch size: ', num_examples, this_batch)     
          finetune(epochs, this_batch, model_copy, optimizer,scheduler, finetune_loader, stop_early, curr_id, setting, curr_iter_setup)
          model_outputs, model_targets = validation(model_copy, test_loaders[i], False)
          model_targets = np.argmax(model_targets, axis=1).reshape(-1, 1)
          model_outputs = np.argmax(model_outputs, axis=1).reshape(-1, 1)
          model_accuracy = metrics.accuracy_score(model_targets, model_outputs)
          
          results_df = pd.DataFrame({'targets':model_targets.squeeze(), 'outputs':model_outputs.squeeze()})
          results_df.to_csv(f'results_{curr_id}_{num_examples}_{datasets[i]}.csv')

          accuracies.append(model_accuracy)
          languages.append(datasets[i])
          num_examples_arr.append(num_examples)
          all_model_names.append(model_name)
          print('accuracy, num, lang, name: ', model_accuracy, num_examples, datasets[i], model_name)
          num_examples_pow += 1
          num_examples = pow(2, num_examples_pow) 
          if cont == 1:
             cont = 2
             break
          if num_examples * 3 > max_examples:
             num_examples = int(max_examples/3)
             cont = 1
         
          
          
        df = pd.DataFrame(list(zip(all_model_names, languages, num_examples_arr, accuracies)), columns=['Model Name', 'Language', 'Num Examples', 'Accuracy']) 
        df.to_csv(f'{curr_id}-{setting}.csv')    


if __name__=="__main__":
    print('Log: Reading command line args\n')

    argp = argparse.ArgumentParser()

    #params
    argp.add_argument('epochs', help='Epochs', type=int)#
    argp.add_argument('true_batch', help='Batch size', type=int)#
    argp.add_argument('batch_size', help='Batch size', type=int)#
    argp.add_argument('lr', help='Learning rate', type=float)#
    argp.add_argument('dropout', help='Dropout', type=float)#
    argp.add_argument('s0', help='Random seed for main', type=int)
    #flags
    argp.add_argument('stop_early', help='For debugging', type=int)
    #storage
    argp.add_argument('id', help='ID', type=str)
    argp.add_argument('setting', help='setting', type=str)
    argp.add_argument('model_type', help='model type', type=str)
    argp.add_argument('base', help='Base model', type=str, default='bert-base-multilingual-cased')
    argp.add_argument('--datasets', nargs='+', help='Dataset for testing the model', type=str, default=None)
    argp.add_argument('--models', nargs='+', help='Model names', type=str, default=None)
    argp.add_argument('--l1',  nargs='+', help='Layers to unfreeze in first model', type=str, default=None)
    argp.add_argument('--iter_setup', help='Balancing batching and iterations', type=int, default=0)

    argp.add_argument('--verify', help='Verify', type=int, default=0)

    argp.add_argument('--test_set', help='Test set', type=str, default="")

    args = argp.parse_args()
    print('Command line args: epochs, true batch, batch size, lr, dropout, seed, stop early, id, setting, model type, base, test dataset, model list:\n')
    print(args.models, args.datasets, args.epochs, args.true_batch, args.batch_size, args.lr, args.dropout, args.base, args.model_type, args.id, args.s0, args.setting, args.stop_early, args.l1, args.iter_setup, args.verify, args.test_set)
    main(args.models, args.datasets, args.epochs, args.true_batch, args.batch_size, args.lr, args.dropout, args.base, args.model_type, args.id, args.s0, args.setting, args.stop_early, args.l1, args.iter_setup, args.verify, args.test_set)

