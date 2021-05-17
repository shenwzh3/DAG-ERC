import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset
from model import *
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import  train_or_eval_model, save_badcase
from dataset import IEMOCAPDataset
from dataloader import get_IEMOCAP_loaders
from transformers import AdamW
import copy

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model,  dataloader, cuda, args, speaker_vocab, label_vocab):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []

    model.eval()

    for data in dataloader:

        features, label, adj,s_mask, s_mask_onehot,lengths, speaker, utterances = data
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()

        log_prob = model(features, adj,s_mask, s_mask_onehot, lengths) # (B, N, C)

        label = label.cpu().numpy().tolist() # (B, N)
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker

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

    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        print('test_accuracy', avg_accuracy)
        print('test_f1', avg_fscore)
        return
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        print('test_accuracy', avg_accuracy)
        print('test_micro_f1', avg_micro_fscore)
        print('test_macro_f1', avg_macro_fscore)
        return

if __name__ == '__main__':

    #path = './saved_models/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', type=str, default='')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='')

    parser.add_argument('--state_dict_file', type=str, default='')

    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--hidden_dim', type = int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod','linear','bilinear', 'rgcn'], help='Feature size.')
    parser.add_argument('--no_rel_attn',  action='store_true', default=False, help='no relation for edges' )

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--windowp', type=int, default=1,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=0,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')


    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global','past'], help='type of nodal attention')

    args = parser.parse_args()
    print(args)
    
    seed_everything()
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()


    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec = get_IEMOCAP_loaders(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
    n_classes = len(label_vocab['itos'])

    print('building model..')
    model = DAGERC_fushion(args, n_classes)


    if torch.cuda.device_count() > 1:
        print('Multi-GPU...........')
        model = nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
    if cuda:
        model.cuda()
    
    state_dict = torch.load(args.state_dict_file)
    model.load_state_dict(state_dict)
    evaluate(model, test_loader, cuda, args, speaker_vocab, label_vocab)