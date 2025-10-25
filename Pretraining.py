# final_deberta_train.py
import os, re, math, gc, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

warnings.filterwarnings('ignore')

# CONFIG
DATA_CSV = "dataset/train.csv"
OUTDIR = "outputs_deberta"
os.makedirs(OUTDIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(OUTDIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
HOLDOUT_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-5
WARMUP_PCT = 0.06
GRAD_ACCUM = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# UTILS
def smape_original(preds_log, trues_log):
    p, t = np.expm1(preds_log), np.expm1(trues_log)
    denom = np.abs(p) + np.abs(t) + 1e-8
    return 100.0 * np.mean(2.0 * np.abs(p - t) / denom)


def repair_and_load_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().replace("\r", "")
    repaired, buf, in_quote = [], "", False
    for line in raw.split("\n"):
        if line.count('"') % 2 == 1:
            if in_quote:
                buf += " " + line.strip()
                repaired.append(buf)
                buf, in_quote = "", False
            else:
                buf, in_quote = line.strip(), True
        elif in_quote:
            buf += " " + line.strip()
        else:
            repaired.append(line.strip())
    if buf:
        repaired.append(buf)
    tmp = os.path.join(OUTDIR, "train_repaired.csv")
    with open(tmp, "w", encoding="utf-8") as fw:
        fw.write("\n".join(repaired))
    df = pd.read_csv(tmp, quotechar='"', sep=",", escapechar="\\", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "price" not in df.columns:
        for c in df.columns:
            if df[c].astype(str).str.match(r"^\d+(\.\d+)?$").sum() > 0.5 * len(df):
                df.rename(columns={c: "price"}, inplace=True)
                break
    df["price"] = pd.to_numeric(df.get("price", np.nan), errors="coerce")
    if "catalog_content" not in df.columns:
        for c in df.columns:
            if df[c].astype(str).str.contains("Item Name|Product Description", na=False).sum() > 0.05 * len(df):
                df.rename(columns={c: "catalog_content"}, inplace=True)
                break
    df = df.dropna(subset=["price", "catalog_content"]).reset_index(drop=True)
    df = df[df["price"] > 0].reset_index(drop=True)
    df["text"] = df["catalog_content"].astype(str)
    df["y_log"] = np.log1p(df["price"].astype(float))
    return df


class TextDataset(Dataset):
    def __init__(self, texts, y, tokenizer, max_len=128):
        self.texts = texts
        self.y = y
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tok(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.y[idx], dtype=torch.float32)
        }


class DebertaRegressor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.regressor.weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.regressor(pooled).squeeze(-1)


def train_deberta(train_texts, train_y, val_texts, val_y, out_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_texts, train_y, tokenizer, max_len=MAX_LEN)
    val_ds = TextDataset(val_texts, val_y, tokenizer, max_len=MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
    model = DebertaRegressor(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = int(math.ceil(len(train_dl) / GRAD_ACCUM) * EPOCHS)
    warmup_steps = int(total_steps * WARMUP_PCT)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_smape = 1e9
    best_state = None
    loss_fn = nn.SmoothL1Loss()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dl, desc=f"DeBERTa Ep{epoch+1}")):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            running_loss += loss.item() * GRAD_ACCUM

        # Validation
        val_preds, val_trues = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    preds = model(input_ids, attention_mask)
                val_preds.append(preds.detach().cpu().numpy())
                val_trues.append(labels.detach().cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        smape = smape_original(val_preds, val_trues)
        print(f"Epoch {epoch+1} | SMAPE: {smape:.4f} | Train Loss: {running_loss/len(train_dl):.4f}")
        if smape < best_smape:
            best_smape = smape
            best_state = model.state_dict()
            torch.save(best_state, out_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Validation SMAPE: {best_smape:.4f}")
    return model


# PREDICTION
def model_predict(model, texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = TextDataset(texts, np.zeros(len(texts)), tokenizer, max_len=MAX_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(input_ids, attention_mask)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds)


def main():
    df = repair_and_load_csv(DATA_CSV)
    train_df, val_df = train_test_split(df, test_size=HOLDOUT_RATIO, random_state=SEED)
    model_path = os.path.join(CHECKPOINT_DIR, 'deberta_best.pt')
    model = train_deberta(
        train_df['text'].tolist(),
        train_df['y_log'].values,
        val_df['text'].tolist(),
        val_df['y_log'].values,
        model_path
    )
    preds = model_predict(model, val_df['text'].tolist())
    smape_val = smape_original(preds, val_df['y_log'].values)
    print(f"Final Validation SMAPE: {smape_val:.4f}")
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()