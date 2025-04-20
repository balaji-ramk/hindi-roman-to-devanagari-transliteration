import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, S, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class SmallTransformer(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size,
        d_model=128, nhead=2, num_layers=4, dim_feedforward=512, dropout=0.1
    ):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        # src: [B, S], tgt: [B, T]
        device = src.device
        src_emb = self.pos_enc(self.src_embed(src))  # [B, S, D]
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))  # [B, T, D]
        src_emb = src_emb.transpose(0, 1)  # [S, B, D]
        tgt_emb = tgt_emb.transpose(0, 1)  # [T, B, D]
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0), device)
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [T, B, D]
        logits = self.out_proj(out)  # [T, B, V_tgt]
        return logits.transpose(0, 1)  # [B, T, V_tgt]


class TranslitDataset(Dataset):
    def __init__(self, split, char2idx_src, char2idx_tgt, max_len=64):
        data_files = {
            "train": "data/hin_train.parquet",
            "validation": "data/hin_valid.parquet"
        }
        ds = load_dataset("parquet", data_files=data_files)
        self.pairs = []
        for ex in ds['train']:
            src = ex["english word"]
            tgt = ex["native word"]
            # filter too long or too short
            if 1 < len(src) <= max_len and 1 < len(tgt) <= max_len:
                self.pairs.append((src, tgt))
        self.src2i = char2idx_src
        self.tgt2i = char2idx_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = [1] + [self.src2i[c] for c in src] + [2]
        tgt_ids = [1] + [self.tgt2i[c] for c in tgt] + [2]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    max_src, max_tgt = max(src_lens), max(tgt_lens)
    src_pad = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t

    src_mask = src_pad == 0
    tgt_mask = tgt_pad == 0
    return src_pad, tgt_pad, src_mask, tgt_mask


def build_vocabs(train_samples):
    src_chars = set()
    tgt_chars = set()
    for src, tgt in train_samples:
        src_chars.update(src)
        tgt_chars.update(tgt)

    src2i = {c: i + 3 for i, c in enumerate(sorted(src_chars))}
    tgt2i = {c: i + 3 for i, c in enumerate(sorted(tgt_chars))}
    for d in (src2i, tgt2i):
        d["<pad>"], d["<bos>"], d["<eos>"] = 0, 1, 2
    return src2i, tgt2i


def train(args):
    raw_train = load_dataset("parquet", data_files="data/hin_train.parquet", split="train[:5%]")
    samples = [(ex["english word"], ex["native word"]) for ex in raw_train]
    src2i, tgt2i = build_vocabs(samples)

    train_ds = TranslitDataset("train", src2i, tgt2i, max_len=args.max_len)
    val_ds   = TranslitDataset("validation", src2i, tgt2i, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallTransformer(
        src_vocab_size=len(src2i),
        tgt_vocab_size=len(tgt2i),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt, src_mask, tgt_mask in tqdm(train_loader, desc=f"Train E{epoch}"):
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train:.4f}")


        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for src, tgt, src_mask, tgt_mask in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
                logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
            print(f"Epoch {epoch} Val Loss: {(val_loss/len(val_loader)):.4f}")
        scheduler.step()

    torch.save(model.state_dict(), args.save_path)
    print("Model saved to", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--d_model",    type=int, default=128)
    parser.add_argument("--nhead",      type=int, default=2)
    parser.add_argument("--layers",     type=int, default=4)
    parser.add_argument("--ffn_dim",    type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--max_len",    type=int, default=64)
    parser.add_argument("--save_path",  type=str, default="hin_xlit_small.pt")
    args = parser.parse_args()
    train(args)
