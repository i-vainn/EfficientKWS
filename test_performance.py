import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from IPython.display import clear_output
from collections import defaultdict

from kws.utils import *
from kws.config import *
from kws.data_utils import *
from kws.augmentations import *
from kws.models import *
from kws.trainer import *


def test_preformance(config, model=None):
    seed_everything()

    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config.keyword
    )

    indexes = torch.randperm(len(dataset))
    val_indexes = indexes[int(len(dataset) * 0.8):]

    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)
    val_set = SpeechCommandDataset(csv=val_df)
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size,
        shuffle=False, collate_fn=Collator(),
        num_workers=2, pin_memory=True
    )
    melspec_val = LogMelspec(is_train=False, config=config)

    if not model:
        model = CRNN(config)
    model.to(config.device)

    validation_dict = validation(
        model, val_loader,
        melspec_val, config.device, True
    )

    return validation_dict
