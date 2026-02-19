from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import gpt # Importing the specific architecture

# 1. Setup the API
app = FastAPI(title="Financial GPT API", version="1.0")

# 2. Rebuild the Tokenizer
# Load the dataset to get the unique characters for the mapping
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    print("Error: input.txt not found.")

# 3. Load the Model and Weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

model = gpt.BigramLanguageModel(vocab_size)
try:
    # Load the weights saved earlier
    model.load_state_dict(torch.load('financial_gpt_weights.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode (turns off Dropout)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: financial_gpt_weights.pth not found.")

# 4. Define the Request Body
class GenerateRequest(BaseModel):
    max_tokens: int = 100

# 5. Define the Endpoint
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    if request.max_tokens > 2000:
        raise HTTPException(status_code=400, detail="Max tokens cannot exceed 2000")
    
    # Start with a blank context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    generated_ids = model.generate(context, max_new_tokens=request.max_tokens)
    
    # Decode back to text
    output_text = decode(generated_ids[0].tolist())
    
    return {"generated_text": output_text}

# Run with: uvicorn app:app --reload