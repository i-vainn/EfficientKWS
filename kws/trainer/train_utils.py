import wandb
import torch
import numpy as np
import torch.nn.functional as F
from thop import profile
from time import time
from tqdm.auto import tqdm
from kws.metrics import count_FA_FR, get_au_fa_fr
from kws.utils import get_size_in_megabytes


def train_epoch(model, opt, loader, log_melspec, device, scheduler=None):
    model.train()
    loss_log = []
    for it, (batch, labels) in tqdm(enumerate(loader, 1), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        loss_log.append(loss.item())
        opt.step()

        if it % 20 == 0:
            wandb.log(dict(
                train_loss=np.mean(loss_log[-20:]),
                learning_rate=opt.param_groups[0]['lr']
            ))

        if scheduler:
            scheduler.step()


def distill_loss(student_raw, teacher_logprobs, labels, alpha):
    student_logprobs = F.log_softmax(student_raw, dim=-1)
    l_dst = F.kl_div(student_logprobs, teacher_logprobs, log_target=True)
    l_regular = F.cross_entropy(student_raw, labels)
    return alpha * l_dst + (1 - alpha) * l_regular

def distilled_train_epoch(student_model, teacher_model, opt, loader, log_melspec, device, alpha, scheduler=None):
    student_model.train()
    teacher_model.eval()
    loss_log = []
    for it, (batch, labels) in tqdm(enumerate(loader, 1), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        with torch.inference_mode():
            teacher_raw = teacher_model(batch)
            teacher_logprobs = F.log_softmax(teacher_raw, dim=-1)

        student_raw = student_model(batch)
        loss = distill_loss(student_raw, teacher_logprobs, labels, alpha)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
        loss_log.append(loss.item())
        opt.step()

        if it % 20 == 0:
            wandb.log(dict(
                train_loss=np.mean(loss_log[-20:]),
                learning_rate=opt.param_groups[0]['lr']
            ))

        if scheduler:
            scheduler.step()


def validation(model, loader, log_melspec, device, do_profile=False):
    model.eval()

    val_losses, accs, FAs, FRs = [], [], [], []
    all_probs, all_labels = [], []
    all_times, macs = [], []
    with torch.inference_mode():
        for _, (batch, labels) in tqdm(enumerate(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = log_melspec(batch)

            start_time = time()
            output = model(batch)
            all_times.append(time() - start_time)

            if do_profile:
                res, _ = profile(model, (batch,), verbose=False)
                macs.append(res)

            # we need probabilities so we use softmax & CE separately
            probs = F.softmax(output, dim=-1)
            loss = F.cross_entropy(output, labels)

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            all_probs.append(probs[:, 1].cpu())
            all_labels.append(labels.cpu())
            val_losses.append(loss.item())
            accs.append(
                torch.sum(argmax_probs == labels).item() /  # ???
                torch.numel(argmax_probs)
            )
            FA, FR = count_FA_FR(argmax_probs, labels)
            FAs.append(FA)
            FRs.append(FR)

    # area under FA/FR curve for whole loader
    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    if not do_profile:
        return au_fa_fr

    res = dict(
        memory=get_size_in_megabytes(model),
        au_fa_fr=au_fa_fr,
        time=sum(all_times) / max(1, len(all_times)),
        MACs=sum(macs) / max(1, len(macs)),
        num_params=sum(x.numel() for x in model.parameters())
    )

    return res
    