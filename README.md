# Towards Long Context Hallucination Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/pdf/2504.19457)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.43+-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/Licenssse-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of the paper **"Towards Long Context Hallucination Detection"** by Siyi Liu et al. (2024). The work introduces a novel architecture that enables pre-trained encoder models like BERT to process long contexts and effectively detect contextual hallucinations through a decomposition and aggregation mechanism.

## 📰 Abstract

Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. However, they are prone to contextual hallucination, generating information that is either unsubstantiated or contradictory to the given context. Although many studies have investigated contextual hallucinations in LLMs, addressing them in long-context inputs remains an open problem. In this work, we take an initial step toward solving this problem by constructing a dataset specifically designed for long-context hallucination detection. Furthermore, we propose a novel architecture that enables pre-trained encoder models, such as BERT, to process long contexts and effectively detect contextual hallucinations through a decomposition and aggregation mechanism.

## 🏗️ Architecture Overview

Our approach consists of three main components:

1. **Decomposition**: Long input contexts and responses are split into smaller, manageable chunks
2. **Encoding**: Each chunk is processed using a backbone encoder model (e.g., RoBERTa-large)  
3. **Aggregation**: Chunk-level representations are combined through a learned attention mechanism to create holistic representations for hallucination detection

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.43+
- Accelerate 0.33+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/amazon-science/long-context-hallucination-detection.git
cd long-context-hallucination-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The dataset is constructed from BookSum, introducing contextual hallucinations using a prompting workflow. The data contains:

- **Training set**: 5,653 examples (51% hallucinations)
- **Test set**: 1,142 examples (50% hallucinations)

We modified the original data by collecting their chapter-level texts, and then prompting GPT-4o to introduce hallucinations to it
by adding/modifying a sentence there (the exact prompt can be found in Appendic C in our paper).

### Data Format

Each example contains:
- `context`: Long document text (book chapters)
- `response`: Summary text that may contain hallucinations
- `labels`: Binary labels (0=faithful, 1=hallucination)

Example:
```json
{
  "context": "Long chapter text from a book...",
  "response": "Summary that may contain hallucinations...",
  "labels": 0,
  "book_id": "1232",
  "book_title": "The Prince",
  "chapter_id": "section 1: chapters 1-3"
}
```

## 🔧 Usage

### Training

To train the model with default hyperparameters:

```bash
bash src/train.sh
```

Or run training with custom parameters:

```bash
accelerate launch src/train.py \
  --model_name_or_path FacebookAI/roberta-large \
  --training_data_path data/train_all.json \
  --testing_data_path data/test_all.json \
  --split \
  --chunk_size 256 \
  --num_chunks1 32 \
  --num_chunks2 8 \
  --attention_encoder \
  --pad_last \
  --split_inputs \
  --output_dir ./outputs
```

To reproduce the model, use the default hyperparameters.

### Evaluation

To evaluate a trained model:

```bash
python src/eval.py \
  --model_name_or_path path/to/trained/model \
  --testing_data_path data/test_all.json \
  --split \
  --chunk_size 256 \
  --num_chunks1 32 \
  --num_chunks2 8 \
  --attention_encoder \
  --pad_last \
  --split_inputs
```

### Key Arguments

- `--split`: Enable chunk splitting for long contexts
- `--attention_encoder`: Use attention aggregation (vs mean pooling)
- `--pad_last`: Pad at sequence end rather than each chunk
- `--split_inputs`: Separate context and response chunks
- `--chunk_size`: Size of each chunk (default: 256)
- `--num_chunks1`: Number of context chunks (default: 32)  
- `--num_chunks2`: Number of response chunks (default: 8)
- `--maximal_text_length`: Maximum total text length (default: 8192)



## 🔍 Model Architecture Details

### Chunk Processing
- Input texts are tokenized and split into overlapping chunks
- Special tokens ([CLS], [SEP]) are added to each chunk
- Chunks are processed in parallel through the backbone encoder

### Attention Aggregation  
- Chunk-level [CLS] representations are fed to a single-layer RoBERTa attention module
- A learnable [CLS] token aggregates information across all chunks
- Final classification is performed on the aggregated representation


## 📁 Repository Structure

```
├── data/                          # Dataset files
│   ├── train_all.json            # Training data
│   └── test_all.json             # Test data
├── src/                          # Source code
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   ├── split_chunks.py           # Text chunking utilities
│   └── train.sh                  # Training shell script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔬 Hallucination Types

The dataset includes two types of contextual hallucinations:

1. **Contradictory Information**: Content that directly contradicts the source context
2. **Unsubstantiated Information**: New information not present or implied in the context

## 🤝 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{liu2024towards,
  title={Towards Long Context Hallucination Detection},
  author={Liu, Siyi and Halder, Kishaloy and Qi, Zheng and Xiao, Wei and Pappas, Nikolaos and Htut, Phu Mon and John, Neha Anna and Benajiba, Yassine and Roth, Dan},
  journal={arXiv preprint arXiv:2504.19457},
  year={2024}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BookSum dataset authors for providing the foundation data
- The research was conducted at AWS AI Labs and University of Pennsylvania

## 🐛 Issues

If you encounter any issues or have questions, please open an issue on GitHub or contact the authors.