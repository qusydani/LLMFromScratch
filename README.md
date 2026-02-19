# LLMFromScratch

<img width="865" height="798" alt="image" src="https://github.com/user-attachments/assets/80a5c117-1e1d-451b-9f80-06a85c995f5e" />

## Overview
Financial GPT is a decoder-only Transformer model trained from scratch on a dataset of S&P 500 earnings call transcripts. It generates context-aware financial text, mimicking the structure and vocabulary of corporate earnings reports.
This project **implements the Transformer architecture purely in PyTorch**, including self-attention mechanisms, feed-forward networks, and residual connections.

The model is deployed as a full-stack application with a **FastAPI** backend for inference and a **Streamlit** frontend for user interaction.

## Key Features
**Custom Transformer Architecture:** Built MultiHeadAttention, LayerNorm, and FeedForward blocks from scratch.

**10 Million Parameters:** Scaled up from a toy model to a standard small-scale LLM (n_embd=384, n_head=6, n_layer=6).

**Inference API:** Decoupled architecture serving the model via a REST API using FastAPI.

**Interactive UI:** User-friendly web interface built with Streamlit.

**GPU Accelerated:** Optimized for CUDA-enabled training and inference.

## Key Concepts Learnt

This project involved building a Decoder-Only Transformer architecture from the ground up, to implement the core mathematical operations of Generative AI. 

**Self-Attention Mechanism (The "Communication" Phase):**
- Implemented Scaled Dot-Product Attention to allow tokens to dynamically weigh the importance of past context.
- Engineered Multi-Head Attention to parallelize this process, allowing the model to capture different types of relationships simultaneously across 6 independent heads.
- Applied Causal Masking (using a lower-triangular matrix) to ensure the model cannot "cheat" by seeing future tokens during training, preserving the auto-regressive property.

**Position-wise Feed-Forward Networks (The "Computation" Phase):**
- Developed the MLP blocks that follow attention layers, consisting of two linear transformations with a ReLU activation in between.
- Implemented the standard expansion factor of 4x, creating a high-dimensional space for the model to process and "reason" about the information gathered during attention.

**Residual Connections & Layer Normalization:**
- Integrated Skip Connections around both attention and feed-forward blocks to prevent vanishing gradients, allowing for the training of deeper networks.
- Applied Layer Normalization before each block to stabilize training dynamics by normalizing input distributions and mitigating internal covariate shift.

**Learnable Positional Embeddings:**
- Unlike RNNs, Transformers process data in parallel and lack inherent notions of order. I implemented a learnable positional embedding table (block_size, n_embd) that is added to the token embeddings, giving the model awareness of sequence order and relative positioning.

**Optimization & Training Dynamics:**
- Utilized the AdamW optimizer for weight decay handling.Implemented Dropout (0.2) throughout the network to act as regularization and prevent overfitting on the financial dataset. Managed tensor shapes carefully to avoid broadcasting errors, specifically aligning head_size with n_embd // n_head.


## Tech Stack
**Deep Learning:** PyTorch (CUDA 12.x)

**Backend:** FastAPI, Uvicorn

**Frontend:** Streamlit

**Data Processing:** Python (Native)

**Version Control:** Git

## Model Architecture
The model follows a standard GPT-style decoder-only architecture:
- **Embedding Dimension (n_embd):** 384
- **Number of Heads (n_head):** 6
- **Number of Layers (n_layer):** 6
- **Block Size (Context Window):** 8 tokens
- **Dropout:** 0.2
- **Parameter Count:** ~10.7 Million

## Installation & Setup
**1. Clone Repository**

```
git clone https://github.com/qusydani/LLMFromScratch.git
cd financial-gpt
```

**2. Create Virtual Environment**

```
python -m venv .venv
.venv\Scripts\activate
```

**4. Install Dependencies**

```
pip install -r requirements.txt
```

## Usage
**1. Train the Model from scratch (with input.txt)**

```
python v2.py
```

This will generate financial_gpt_weights.pth after training.

**2. Start the FastAPI backend**

```
uvicorn app:app --reload
```

**3. Launch the Streamlit interface**

```
streamlit run frontend.py
```
