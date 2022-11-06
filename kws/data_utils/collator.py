import torch
from torch.nn.utils.rnn import pad_sequence


class Collator:
    
    def __call__(self, data):
        wavs = []
        labels = []    

        for el in data:
            wavs.append(el['wav'])
            labels.append(el['label'])

        wavs = pad_sequence(wavs, batch_first=True)    
        labels = torch.Tensor(labels).long()
        
        return wavs, labels