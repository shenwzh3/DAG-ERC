# DAG-ERC
Pytorch code for ACL-IJCNLP 2021 accepted paper "Directed Acyclic Graph Network for Conversational Emotion Recognition"

## Requirements
* Python 3.6
* PyTorch 1.6.0
* [Transformers](https://github.com/huggingface/transformers) 3.5.1
* CUDA 10.1

## Preparation

### Datasets and Utterance Feature
You can download the dataset and extracted utterance feature from 
https://drive.google.com/file/d/1R5K_2PlZ3p3RFQ1Ycgmo3TgxvYBzptQG/view?usp=sharing
or https://pan.baidu.com/s/1H_LXQbDCfbWlwG1KvzNW6Q 提取码 c9vk

## Training
You can train the models with the following codes:

For IEMOCAP: 
`python run.py --dataset IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.2`

For MELD: 
`python run.py --dataset MELD --lr 0.00001 --batch_size 64 --epochs 70 --dropout 0.1`

For DailyDialog: 
`python run.py --dataset EmoryNLP --lr 0.00005 --batch_size 32 --epochs 100 --dropout 0.3`

For EmoryNLP: 
`python run.py --dataset DailyDialog --gnn_layers 3 --lr 0.00002 --batch_size 64 --epochs 50 --dropout 0.3`
