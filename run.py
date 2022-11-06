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

def run(config):
    seed_everything()

    dataset = SpeechCommandDataset(
        path2dir='speech_commands', keywords=config.keyword
    )

    # train-val split
    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * 0.8)]
    val_indexes = indexes[int(len(dataset) * 0.8):]

    train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)
    train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
    val_set = SpeechCommandDataset(csv=val_df)

    # defining dataloaders
    # Here we are obliged to use shuffle=False because of our sampler with randomness inside.
    train_sampler = get_sampler(train_set.csv['label'].values)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size,
        shuffle=False, collate_fn=Collator(),
        sampler=train_sampler,
        num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=config.batch_size,
        shuffle=False, collate_fn=Collator(),
        num_workers=2, pin_memory=True
    )

    melspec_train = LogMelspec(is_train=True, config=config)
    melspec_val = LogMelspec(is_train=False, config=config)

    model = StreamCRNN(config)
    model.to(config.device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    history = defaultdict(list)

    with Timer('Train') as timer:
        for epoch in range(config.num_epochs):

            train_epoch(
                model, opt, train_loader,
                melspec_train, config.device
            )
            au_fa_fr = validation(
                model, val_loader,
                melspec_val, config.device
            )
            history['val_metric'].append(au_fa_fr)

            clear_output()
            plt.plot(history['val_metric'])
            plt.ylabel('Metric')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()

            print('END OF EPOCH', epoch)

            # m = torch.jit.script(model)
            # torch.jit.save(m, f'baseline_model_{epoch}.pth')
            torch.save(model.state_dict(), f'normal_model_{epoch}.pth')

