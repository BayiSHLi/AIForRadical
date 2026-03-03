import torch
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("CUDA:", torch.cuda.is_available())
print("FAISS GPU:", faiss.get_num_gpus())

embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
print("Embedding loaded")
