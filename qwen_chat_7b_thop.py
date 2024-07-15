from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from zeus.monitor import ZeusMonitor
from thop import profile
from thop import clever_format

device = torch.device("cuda")
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
model.to(device)

output_token_num = 0

while output_token_num < 200:
    macs, params, responce, history = profile(tokenizer, model, inputs="please generate some content about some scientist with at least 500 words")
    output_token_num = len(tokenizer.encode(responce))

macs, params = clever_format([macs, params], "%.3f")
print("macs: ", macs)
print("params: ", params)