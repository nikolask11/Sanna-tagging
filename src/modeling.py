import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

MAX_LEN = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(sentences, tokenizer, label2id):
    feats = []
    for sent in sentences:
        enc = tokenizer(sent["tokens"], is_split_into_words=True,
                        truncation=True, max_length=MAX_LEN)
        word_ids = enc.word_ids()
        labels, first, wids, seen = [], [], [], set()
        for wid in word_ids:
            if wid is None or wid in seen:
                labels.append(-100)
                first.append(0)
                wids.append(-1)
            else:
                seen.add(wid)
                tag = sent["upos"][wid]
                labels.append(label2id.get(tag, -100) if tag is not None else -100)
                first.append(1)
                wids.append(wid)
        feats.append({"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"],
                      "labels": labels, "first": first, "word_ids": wids})
    return feats


def _collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {}
        for key, pad in (("input_ids", pad_id), ("attention_mask", 0),
                         ("labels", -100), ("first", 0), ("word_ids", -1)):
            out[key] = torch.tensor(
                [b[key] + [pad] * (n - len(b[key])) for b in batch], dtype=torch.long)
        return out
    return fn


def train_model(model_name, train_feats, num_labels, seed, pad_id,
                target_steps=500, batch_size=16, lr=3e-5, max_steps=None):
    from transformers import AutoModelForTokenClassification
    set_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels)
    device = get_device()
    model.to(device)
    loader = DataLoader(train_feats, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate(pad_id),
                        generator=torch.Generator().manual_seed(seed))
    spe = max(1, len(loader))
    epochs = min(100, max(3, math.ceil(target_steps / spe)))
    total = epochs * spe if max_steps is None else max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    step = 0
    while step < total:
        for batch in loader:
            if step >= total:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
    model.eval()
    return model


@torch.no_grad()
def predict(model, feats, id2label, pad_id, batch_size=64):
    device = next(model.parameters()).device
    loader = DataLoader(feats, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate(pad_id))
    out = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        conf, pred = conf.cpu().numpy(), pred.cpu().numpy()
        first = batch["first"].numpy()
        wids = batch["word_ids"].numpy()
        for i in range(ids.shape[0]):
            sent = []
            for j in np.nonzero(first[i])[0]:
                sent.append((int(wids[i][j]), id2label[int(pred[i][j])], float(conf[i][j])))
            out.append(sent)
    return out


def evaluate(model, feats, sentences, id2label, pad_id):
    preds = predict(model, feats, id2label, pad_id)
    total = correct = 0
    tp, fp, fn = {}, {}, {}
    for sent, pred in zip(sentences, preds):
        for widx, ptag, _ in pred:
            gold = sent["upos"][widx]
            if gold is None:
                continue
            total += 1
            if ptag == gold:
                correct += 1
                tp[gold] = tp.get(gold, 0) + 1
            else:
                fp[ptag] = fp.get(ptag, 0) + 1
                fn[gold] = fn.get(gold, 0) + 1
    per_tag = {}
    for tag in sorted(set(tp) | set(fp) | set(fn)):
        p = tp.get(tag, 0) / max(1, tp.get(tag, 0) + fp.get(tag, 0))
        r = tp.get(tag, 0) / max(1, tp.get(tag, 0) + fn.get(tag, 0))
        per_tag[tag] = round(2 * p * r / max(1e-9, p + r), 4)
    return correct / max(1, total), per_tag
