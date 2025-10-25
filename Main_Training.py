import os, re, math, gc, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

warnings.filterwarnings("ignore")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 60
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTDIR = "outputs_deberta"
os.makedirs(os.path.join(OUTDIR, "checkpoints"), exist_ok=True)
BASE_CHECKPOINT = "outputs_deberta_checkpoints_deberta_best.pt"
BEST_MODEL_PATH = os.path.join(OUTDIR, "checkpoints", "deberta_fused_best_smape.pt")

TRAIN_CSV = "train_processed.csv"  


def smape_original(preds_log, trues_log):
    p, t = np.expm1(preds_log), np.expm1(trues_log)
    denom = np.abs(p) + np.abs(t) + 1e-8
    return 100.0 * np.mean(2.0 * np.abs(p - t) / denom)


FEATURE_COLS = [
    "is_organic", "is_gourmet", "is_gluten_free",
    "value", "num_sentences", "num_words",
    "has_special_chars", "uppercase_ratio", "is_bulk"
]

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["catalog_content"].astype(str).tolist()
        self.targets = df["log_price"].values if "log_price" in df else None
        self.features = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tok(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': torch.tensor(self.features[idx], dtype=torch.float)
        }
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item


class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim, feat_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, text_dim)
        self.key_proj = nn.Linear(feat_dim, text_dim)
        self.value_proj = nn.Linear(feat_dim, text_dim)
        self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(text_dim)
        self.ff = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, text_dim))

    def forward(self, text_emb, feat_emb):
        q = self.query_proj(text_emb).unsqueeze(1)
        k = self.key_proj(feat_emb).unsqueeze(1)
        v = self.value_proj(feat_emb).unsqueeze(1)
        attn_output, _ = self.attn(q, k, v)
        out = self.norm(text_emb + attn_output.squeeze(1))
        return out + self.ff(out)

class DebertaWithFeatures(nn.Module):
    def __init__(self, model_name=MODEL_NAME, base_checkpoint=None, num_features=len(FEATURE_COLS)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.feat_embed = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.cross_attn = CrossAttentionBlock(hidden_size, 128)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 1)

        if base_checkpoint and os.path.exists(base_checkpoint):
            print(f"🔹 Loading pretrained DeBERTa weights from {base_checkpoint}")
            base_state = torch.load(base_checkpoint, map_location="cpu")
            filtered = {k.replace("model.", "backbone."): v for k, v in base_state.items() if "regressor" not in k}
            self.load_state_dict(filtered, strict=False)
            print("DeBERTa backbone loaded successfully.")

    def forward(self, input_ids, attention_mask, features):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_emb = out.last_hidden_state[:, 0, :]
        feat_emb = self.feat_embed(features)
        fused = self.cross_attn(cls_emb, feat_emb)
        fused = self.dropout(fused)
        return self.regressor(fused).squeeze(-1)


def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        features = batch['features'].to(DEVICE)
        targets = batch['target'].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(input_ids, attention_mask, features)
            loss = loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            outputs = model(input_ids, attention_mask, features)
            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)



def main():
    print("Loading dataset...")
    df = pd.read_csv(TRAIN_CSV)
    df["log_price"] = np.log1p(df["price"])

    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_df, tokenizer, MAX_LEN)
    val_ds = TextDataset(val_df, tokenizer, MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Initializing model...")
    model = DebertaWithFeatures(model_name=MODEL_NAME, base_checkpoint=BASE_CHECKPOINT).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_smape = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, loss_fn)
        val_preds, val_trues = validate_epoch(model, val_dl)
        smape = smape_original(val_preds, val_trues)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | SMAPE: {smape:.4f}")

        if smape < best_smape:
            best_smape = smape
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model (SMAPE: {best_smape:.4f}) to {BEST_MODEL_PATH}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best SMAPE: {best_smape:.4f}")

if __name__ == "__main__":
    main()
