import torch
import wandb
import dataclasses
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from IPython.display import clear_output
from collections import defaultdict

from kws.utils import *
from kws.config import *
from kws.data_utils import *
from kws.augmentations import *
from kws.models import *
from kws.trainer import train_epoch, distilled_train_epoch, validation


class Trainer:

    def __init__(self, config):
        self.config = config
        seed_everything()
        self.setup_data()
        wandb.init(
            project='EfficientKWS',
            entity='i_vainn',
            config=dataclasses.asdict(self.config),
        )


    def setup_data(self):
        dataset = SpeechCommandDataset(
            path2dir='speech_commands', keywords=self.config.keyword
        )

        # train-val split
        indexes = torch.randperm(len(dataset))
        train_indexes = indexes[:int(len(dataset) * 0.8)]
        val_indexes = indexes[int(len(dataset) * 0.8):]

        train_df = dataset.csv.iloc[train_indexes].reset_index(drop=True)
        val_df = dataset.csv.iloc[val_indexes].reset_index(drop=True)
        self.train_set = SpeechCommandDataset(csv=train_df, transform=AugsCreation())
        val_set = SpeechCommandDataset(csv=val_df)

        # defining dataloaders
        # Here we are obliged to use shuffle=False because of our sampler with randomness inside.
        train_sampler = get_sampler(self.train_set.csv['label'].values)

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=Collator(),
            sampler=train_sampler,
            num_workers=2, pin_memory=True
        )

        self.val_loader = DataLoader(
            val_set, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=Collator(),
            num_workers=2, pin_memory=True
        )

        self.melspec_train = LogMelspec(is_train=True, config=self.config)
        self.melspec_val = LogMelspec(is_train=False, config=self.config)


    def train(self):
        model = StreamCRNN(self.config)
        model.to(self.config.device)
        best_score = 1.
        opt = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=self.config.learning_rate,
            total_steps=self.config.num_epochs * (len(self.train_set) + 1) // self.config.batch_size
        )
        history = defaultdict(list)

        for epoch in range(self.config.num_epochs):

            train_epoch(
                model, opt, self.train_loader,
                self.melspec_train, self.config.device,
                scheduler=scheduler
            )
            au_fa_fr = validation(
                model, self.val_loader,
                self.melspec_val, self.config.device
            )
            history['val_metric'].append(au_fa_fr)
            wandb.log(dict(
                val_au_fa_fr=au_fa_fr,
            ))
            self.plot_history(history)
            print('END OF EPOCH', epoch)

            if au_fa_fr < best_score:
                best_score = au_fa_fr
                path = 'checkpoints/{}-{}-{:.7f}_model.pth'.format(wandb.run.name, epoch, au_fa_fr)
                torch.save(model.state_dict(), path)
                wandb.save(path)
            

        val_stats = validation(
            model, self.val_loader,
            self.melspec_val, self.config.device, True
        )
        wandb.log(
            val_stats
        )
        wandb.run.finish()


    def train_distilled(self, small_config, teacher_checkpoint):
        teacher_model = StreamCRNN(self.config)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=self.config.device))
        student_model = StreamCRNN(small_config)
        teacher_model.to(self.config.device)
        student_model.to(self.config.device)
        best_score = 1.
        opt = torch.optim.Adam(
            student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=self.config.learning_rate,
            total_steps=self.config.num_epochs * (len(self.train_set) + 1) // self.config.batch_size
        )
        history = defaultdict(list)

        for epoch in range(small_config.num_epochs):

            distilled_train_epoch(
                student_model, teacher_model, opt, self.train_loader,
                self.melspec_train, self.config.device, 
                small_config.alpha, scheduler=scheduler
            )
            au_fa_fr = validation(
                student_model, self.val_loader,
                self.melspec_val, self.config.device
            )
            history['val_metric'].append(au_fa_fr)
            wandb.log(dict(
                val_au_fa_fr=au_fa_fr,
            ))
            self.plot_history(history)
            print('END OF EPOCH', epoch)
            if au_fa_fr < best_score:
                best_score = au_fa_fr
                path = 'checkpoints/{}-{}-{:.7f}_distill_model.pth'.format(wandb.run.name, epoch, au_fa_fr)
                torch.save(student_model.state_dict(), path)
                wandb.save(path)
            
        val_stats = validation(
            student_model, self.val_loader,
            self.melspec_val, self.config.device, True
        )
        wandb.log(
            val_stats
        )
        wandb.run.finish()


    def plot_history(self, history):
        clear_output()
        plt.plot(history['val_metric'])
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.grid()
        plt.show()