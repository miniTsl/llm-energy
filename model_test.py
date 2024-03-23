from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from zeus.monitor import ZeusMonitor

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()


print(model.transformer.h)
print(len(model.transformer.h))
print(model.transformer.h[0])