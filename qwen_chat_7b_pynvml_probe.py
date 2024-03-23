from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from zeus.monitor import ZeusMonitor

# 从命令行中读取参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hook", type=bool, default=False)
parser.add_argument("--module", type=str, default=None)
args = parser.parse_args()

# 需要减去GPU的基础功耗，静置时GPU大概也有22-30W的功耗
device = torch.device("cuda")
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], log_file="probe.log", approx_instant_energy=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
# define hook function
def forward_hook(module, input, output):
    new_output = module.forward(input[0])  

result_file_name_prefix = "Qwen_7b_chat"
module_name = args.module
hook = args.hook

if hook:
    result_file_name = result_file_name_prefix + "_" + module_name + ".txt"
    hook_handle = []
    # register the hook
    if args.module == "lm_head":
        hook_handle.append(model.lm_head.register_forward_hook(forward_hook))
    elif args.module == "transformer_RNSNorm":
        hook_handle.append(model.transformer.ln_f.register_forward_hook(forward_hook))
    elif args.module == "transformer_Embedding":
        hook_handle.append(model.transformer.wte.register_forward_hook(forward_hook))
    elif args.module == "transformer_Dropout":
        hook_handle.append(model.transformer.drop.register_forward_hook(forward_hook))
    elif args.module == "transformer_RotaryEmbedding":
        hook_handle.append(model.transformer.rotary_emb.register_forward_hook(forward_hook))
    elif args.module == "transformer_h_QWenBlock":
        for i in range(len(model.transformer.h)):
            hook_handle.append(model.transformer.h[i].register_forward_hook(forward_hook))
    elif args.module == "transformer_h_QWenBlock_RMSNorm":
        for i in range(len(model.transformer.h)):
            hook_handle.append(model.transformer.h[i].ln_1.register_forward_hook(forward_hook))
            hook_handle.append(model.transformer.h[i].ln_2.register_forward_hook(forward_hook))
    elif args.module == "transformer_h_QWenBlock_QWenAttention":
        for i in range(len(model.transformer.h)):
            hook_handle.append(model.transformer.h[i].attn.register_forward_hook(forward_hook))
    elif args.module == "transformer_h_QWenBlock_QWenMLP":
        for i in range(len(model.transformer.h)):
            hook_handle.append(model.transformer.h[i].mlp.register_forward_hook(forward_hook))
    # todo: 32 * QWenBlock
else:
    result_file_name = result_file_name_prefix + ".txt"

result_file = open(result_file_name, "w")




# monitor.begin_window("init_tokenizer")

# init_tokenizer_eres = monitor.end_window("init_tokenizer")
# print("init_tokenizer time: ", init_tokenizer_eres.time)
# print("init_tokenizer energy: ", init_tokenizer_eres.total_energy)
# # write the result to the file
# result_file.write("init_tokenizer time: " + str(init_tokenizer_eres.time) + "\n")
# result_file.write("init_tokenizer energy: " + str(init_tokenizer_eres.total_energy) + "\n")
# result_file.write("\n")

# monitor.begin_window("init_model")
model = model.to(device)
# init_model_eres = monitor.end_window("init_model")
# print("init_model time: ", init_model_eres.time)
# print("init_model energy: ", init_model_eres.total_energy)
# # write the result to the file
# result_file.write("init_model time: " + str(init_model_eres.time) + "\n")
# result_file.write("init_model energy: " + str(init_model_eres.total_energy) + "\n")
# result_file.write("\n")

inference_token_nums = []
inference_times = []
inference_energies = []
iter_num = 0
warmup_num = 20
while True:
    monitor.begin_window(f"generate_str_{iter_num}")
    response, history = model.chat(tokenizer, "please generate some content about some scientist with at least 500 words", history=None)
    eres = monitor.end_window(f"generate_str_{iter_num}")
    output_token_num = len(tokenizer.encode(response))

    if output_token_num != 200:
        continue
    elif warmup_num > 0:
        warmup_num -= 1
        print("### warmup No." + str(20 - warmup_num))
        print("QWEN chat output token num: ", output_token_num)
        print("QWEN chat time: ", eres.time)
        print("QWEN chat energy: ", eres.total_energy)
        print("\n")
    else:
        iter_num += 1
        inference_token_nums.append(output_token_num)    
        # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
        inference_times.append(eres.time)
        inference_energies.append(eres.total_energy)
        print("QWEN Chat No." + str(iter_num))
        print("QWEN chat output token num: ", output_token_num)
        print("QWEN chat time: ", eres.time)
        print("QWEN chat energy: ", eres.total_energy)
        print("response: ", response)
        print("\n")
        # write the result to the file
        result_file.write("QWEN chat No." + str(iter_num) + "\n")
        result_file.write("QWEN chat output token num: " + str(output_token_num) + "\n")
        result_file.write("QWEN chat time: " + str(eres.time) + "\n")
        result_file.write("QWEN chat energy: " + str(eres.total_energy) + "\n")
        result_file.write("response: " + response + "\n")
        result_file.write("\n")
        if iter_num == 200:
            break


avg_token_num = sum(inference_token_nums) / len(inference_token_nums)
avg_time = sum(inference_times) / len(inference_times)
avg_energy = sum(inference_energies) / len(inference_energies)
print("avg_token_num: ", avg_token_num)
print("avg_time: ", avg_time)
print("avg_energy: ", avg_energy)
# write the result to the file
result_file.write("avg_token_num: " + str(avg_token_num) + "\n")
result_file.write("avg_time: " + str(avg_time) + "\n")
result_file.write("avg_energy: " + str(avg_energy) + "\n")

if hook:
    for hook_handle in hook_handle:
        hook_handle.remove()

result_file.close()
