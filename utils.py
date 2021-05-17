import numpy as np
import torch

def person_embed(speaker_ids, person_vec):
    '''

    :param speaker_ids: torch.Tensor ( T, B)
    :param person_vec: numpy array (num_speakers, 100)
    :return:
        speaker_vec: torch.Tensor (T, B, D)
    '''
    speaker_vec = []
    for t in speaker_ids:
        speaker_vec.append([person_vec[int(i)].tolist() if i != -1 else [0] * 100 for i in t])
    speaker_vec = torch.FloatTensor(speaker_vec)
    return speaker_vec