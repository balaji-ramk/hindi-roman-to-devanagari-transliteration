# Hindi - Roman to देवनागरी Transliteration

## Acknowledgements
Greatly inspired by AI4Bharat's IndicXlit model (cited below). I tried replicating their architecture, but with a slightly simpler approach, reducing number of hidden layers, etc. to lower the parameter count. I also tried to do this blind, only referenced their paper for the theory and explanation, and tried to replicate the training and inference code on my own. This was largely a learning experiment, and I suggest to most people that they use [AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit) for any serious use-cases.

### Key Differences
| Aspect | My Model | AI4Bharat's Model | Justification |
|-------|----------|------------------|---------------|
| Architecture | Simpler, fewer layers | Larger, more complex | Wanted to reduce parameters for efficiency and learning |
| Language Support | Hindi-only | Multilingual | Focused on Hindi due to hardware limits and simplicity |
| Direction | Roman → देवनागरी only | Bidirectional | Roman → देवनागरी felt more practical as most people are exposed to QWERTY keyboards |

## Use of LLMs
I have tried to minimize the use of LLMs while coding this out, as I wanted this to be a learning endeavour. The primary use of LLMs was in Document Question-Answering (DQA), using a locally run Qwen2.5 Instruct 1M. This is because it makes reading the paper easier to begin with, and lets me have a grasp of a summary before I go into the paper myself.
Further, I also used the same for explaining to me certain tasks, but I tried to replicate the code myself rather than copy-pasting.

Please let me know if you have any advice or thoughts on this, as I would love to leverage LLMs in a way that makes me more efficient, but doesn't hinder the learning process.

## How to Run
1. Clone this repo and move into the directory:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Train the model:
   ```
   python train.py
   ```
   
  Feel free to change any arguments (see [here](#arguments-for-trainpy))
   ```
   python train.py --batch_size 64 --epochs 10 --lr 0.001
   ```
4. Run inference:
   ```
   python inference.py
   ```
   
   Feel free to change any arguments (see [here](#arguments-for-inferencepy))
   ```
   python inference.py --model_path hin_xlit_small.pt --parquet_path data/hin_train.parquet --vocab_dir vocab
   ```

### Arguments for `train.py`

| Argument | Type | Default | Description |
|---------|------|---------|-------------|
| `--batch_size` | int | 64 | batch size for training |
| `--epochs` | int | 10 | number of epochs |
| `--lr` | float | 1e-3 | learning rate |
| `--d_model` | int | 128 | embedding dimension |
| `--nhead` | int | 2 | number of attention heads |
| `--layers` | int | 4 | number of transformer layers |
| `--ffn_dim` | int | 512 | feedforward layer size |
| `--dropout` | float | 0.1 | dropout rate |
| `--max_len` | int | 64 | max token length |
| `--save_path` | str | hin_xlit_small.pt | where to save the model |

### Arguments for `inference.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | hin_xlit_small.pt | path to trained model (same as --save-path in train.py) |
| `--parquet_path` | str | data/hin_train.parquet | path to input data |
| `--vocab_dir` | str | vocab | directory containing vocab.pth (will create this automatically the first time it is run if not present) |

## Results
\[TO BE ADDED]

## Future Work
\[TO BE ADDED]

## References & Citations
- [AI4Bharat IndicXlit (Model)](https://github.com/AI4Bharat/IndicXlit) - Model architecture and training methods greatly inspired from here. [Paper](https://arxiv.org/abs/2205.03018).
- [AI4Bharat Aksharantar (Dataset)](https://huggingface.co/datasets/ai4bharat/Aksharantar) - Used the hindi part of this dataset for training the model. (Downloaded the hin.zip, converted the JSONL files to parquet via Pandas for HuggingFace Dataset library).


