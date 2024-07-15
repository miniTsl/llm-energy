from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from zeus.monitor import ZeusMonitor
import time

# 需要减去GPU的基础功耗? 静置时GPU大概也有22-30W的功耗
device = torch.device("cuda")
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], approx_instant_energy=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
# define hook function
def forward_hook(module, input, output):
    print(module, "timestampe: ", time.time())

model = model.to(device)

while True:
    monitor.begin_window(f"generate_chat")
    response, history = model.chat(tokenizer, "please generate some content about some scientist with at least 500 words", history=None)
    eres = monitor.end_window(f"generate_chat")
    output_token_num = len(tokenizer.encode(response))

    if output_token_num != 200:
        continue
    else:
        print("QWEN chat output token num: ", output_token_num)
        print("QWEN chat time: ", eres.time)
        print("QWEN chat energy: ", eres.total_energy)
        print("response: ", response)
        print("\n")
        break