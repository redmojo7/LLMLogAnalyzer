from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
from transformers import AutoTokenizer
from llama_index.llms.groq import Groq
import os


# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
"""
MAX_NEW_TOKENS = 1024
TEMPRAUTURE = 0.7

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
"""

# LLM model
# Llama3
#model = "llama3-8b-8192"
#model = "llama3-70b-8192"
# Llama3.1
#model = "llama-3.1-8b-instant"
model =  "llama-3.1-70b-versatile"
Settings.llm = Groq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)
"""
Settings.llm = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=MAX_NEW_TOKENS,
    generate_kwargs={"temperature": TEMPRAUTURE, "do_sample": False},
    system_prompt=rag_system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=MODEL_ID,
    model_name=MODEL_ID,
    device_map="auto",
    stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)
"""



#groq_llama3_70b_llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
#groq_llama3_8b_llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
