import os
import argparse
import math
import torch
import torch.nn as nn
from datasets import load_dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class SmallTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=2, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
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
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        device = src.device
        src_emb = self.pos_enc(self.src_embed(src)).transpose(0,1)  # [S,B,D]
        tgt_emb = self.pos_enc(self.tgt_embed(tgt)).transpose(0,1)  # [T,B,D]
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0), device)
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [T,B,D]
        return self.out_proj(out).transpose(0,1)  # [B,T,V_tgt]


def build_vocabs_from_parquet(parquet_path, max_len=64):
    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    src_chars = set(); tgt_chars = set()
    for ex in ds:
        s, t = ex["english word"], ex["native word"]
        if 1 < len(s) <= max_len and 1 < len(t) <= max_len:
            src_chars.update(s)
            tgt_chars.update(t)
    src2i = {c: i+3 for i, c in enumerate(sorted(src_chars))}
    tgt2i = {c: i+3 for i, c in enumerate(sorted(tgt_chars))}
    for d in (src2i, tgt2i):
        d["<pad>"], d["<bos>"], d["<eos>"] = 0, 1, 2
    return src2i, tgt2i

def load_or_build_vocab(parquet_path, vocab_dir="vocab"):
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_path = os.path.join(vocab_dir, "vocab.pth")
    if os.path.exists(vocab_path):
        bundle = torch.load(vocab_path, map_location="cpu")
        print(f"Loaded vocab from {vocab_path}")
        return bundle["src2i"], bundle["tgt2i"]
    else:
        print("Building vocab from Parquet, this may take a moment...")
        src2i, tgt2i = build_vocabs_from_parquet(parquet_path)
        torch.save({"src2i": src2i, "tgt2i": tgt2i}, vocab_path)
        print(f"Saved vocab to {vocab_path}")
        return src2i, tgt2i


def greedy_decode(model, src_ids, src_mask, src2i, i2tgt, max_len=64):
    device = src_ids.device
    ys = torch.tensor([[src2i["<bos>"]]], device=device)
    for _ in range(max_len):
        tgt_mask = ys == src2i["<pad>"]
        out = model(src_ids, ys, src_mask, tgt_mask)
        next_id = int(out[0, -1].argmax())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == src2i["<eos>"]:
            break

    chars = []
    for idx in ys[0].tolist()[1:]:
        if idx == src2i["<eos>"]:
            break
        chars.append(i2tgt.get(idx, ""))
    return "".join(chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   default="hin_xlit_small.pt")
    parser.add_argument("--parquet_path", default="data/hin_train.parquet")
    parser.add_argument("--vocab_dir",    default="vocab")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    src2i, tgt2i = load_or_build_vocab(args.parquet_path, args.vocab_dir)
    i2tgt = {i: c for c, i in tgt2i.items()}

    model = SmallTransformer(
        src_vocab_size=len(src2i),
        tgt_vocab_size=len(tgt2i),
        d_model=128, nhead=2, num_layers=4, dim_feedforward=512, dropout=0.1
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    print("Ready! Type romanized Hindi words (or 'quit'):")
    while True:
        text = input("> ").strip()
        if not text or text.lower() in ("quit", "exit"):
            break
        ids = [src2i.get(c, src2i["<pad>"]) for c in text]
        src_ids = torch.tensor([[src2i["<bos>"]] + ids + [src2i["<eos>"]]],
                               device=args.device)
        src_mask = src_ids == src2i["<pad>"]
        with torch.no_grad():
            out = greedy_decode(model, src_ids, src_mask, src2i, i2tgt)
        print("→", out)
