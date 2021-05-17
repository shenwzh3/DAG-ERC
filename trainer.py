import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from utils import person_embed
from tqdm import tqdm
import json


def train_or_eval_model(model, loss_function, dataloader,epoch, cuda, args, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []


    assert not train or optimizer != None
    if train:
        model.train()
        # dataloader = tqdm(dataloader)
    else:
        model.eval()

    cnt = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        features, label, adj,s_mask, s_mask_onehot,lengths, speakers, utterances = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask = s_mask.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            lengths = lengths.cuda()


        # print(speakers)
        log_prob = model(features, adj, s_mask, s_mask_onehot, lengths) # (B, N, C)
        # print(label)
        loss = loss_function(log_prob.permute(0,2,1), label)


        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist()
        preds += pred
        labels += label
        losses.append(loss.item())


        if train:
            loss_val = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_micro_fscore, avg_macro_fscore


def save_badcase(model,  dataloader, cuda, args, speaker_vocab, label_vocab):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []

    model.eval()

    for data in dataloader:

        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        features, label, adj,s_mask, s_mask_onehot,lengths, speaker, utterances = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()


        # print(speakers)
        log_prob = model(features, adj,s_mask, s_mask_onehot, lengths) # (B, N, C)


        label = label.cpu().numpy().tolist() # (B, N)
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker

        # finished here

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return

    cases = []
    for i,d in enumerate(dialogs):
        case = []
        for j,u in enumerate(d):
            case.append({
                'text': u,
                'speaker': speaker_vocab['itos'][speakers[i][j]],
                'label': label_vocab['itos'][labels[i][j]] if labels[i][j] != -1 else 'none',
                'pred': label_vocab['itos'][preds[i][j]]
            })
        cases.append(case)

    with open('badcase/%s.json'%(args.dataset_name), 'w', encoding='utf-8') as f:
        json.dump(cases,f)

    # print(preds.tolist())
    # print(labels.tolist())
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        print('badcase saved')
        print('test_f1', avg_fscore)
        return
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        print('badcase saved')
        print('test_micro_f1', avg_micro_fscore)
        print('test_macro_f1', avg_macro_fscore)
        return
