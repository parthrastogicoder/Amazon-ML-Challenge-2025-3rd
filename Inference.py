import os, gc, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTDIR = "outputs_deberta"
BEST_MODEL_PATH = "outputs_deberta_checkpoints_deberta_fused_best_smape.pt"
TEST_CSV = "test_processed.csv"
OUTPUT_CSV = os.path.join(OUTDIR, "submission.csv")

FEATURE_COLS = [
    "is_organic", "is_gourmet", "is_gluten_free",
    "value", "num_sentences", "num_words",
    "has_special_chars", "uppercase_ratio", "is_bulk"
]

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["catalog_content"].astype(str).tolist()
        self.features = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tok(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': torch.tensor(self.features[idx], dtype=torch.float)
        }

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
    def __init__(self, model_name=MODEL_NAME, num_features=len(FEATURE_COLS)):
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

    def forward(self, input_ids, attention_mask, features):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_emb = out.last_hidden_state[:, 0, :]
        feat_emb = self.feat_embed(features)
        fused = self.cross_attn(cls_emb, feat_emb)
        fused = self.dropout(fused)
        return self.regressor(fused).squeeze(-1)

def inference():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_ds = TextDataset(test_df, tokenizer, MAX_LEN)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("🧠 Loading model...")
    model = DebertaWithFeatures(model_name=MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded checkpoint from {BEST_MODEL_PATH}")

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Predicting"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            features = batch["features"].to(DEVICE)
            outputs = model(input_ids, attention_mask, features)
            preds.extend(outputs.cpu().numpy())

    test_df["price"] = np.expm1(preds)  
    submission = test_df[["sample_id", "price"]]
    submission.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved predictions to {OUTPUT_CSV}")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    inference()
