from test import qwen_test
import matplotlib.pyplot as plt 

init_tokenizer_eres, init_model_eres, inference_token_nums, inference_times, inference_energies = qwen_test().do_something()

# print
print(f"Init tokenizer consumed {init_tokenizer_eres.time} s and {init_tokenizer_eres.total_energy} J.")
print(f"Init model consumed {init_model_eres.time} s and {init_model_eres.total_energy} J.")
print(f"Total {len(inference_token_nums)} generations.")
for i in range(len(inference_token_nums)):
    print(f"Generation {i}_unsorted consumed {inference_times[i]} s and {inference_energies[i]} J, with {inference_token_nums[i]} tokens.")

# sort the data by token number
inference_token_nums, inference_times, inference_energies = zip(*sorted(zip(inference_token_nums, inference_times, inference_energies)))
# if there are multiple generations with the same token number, average their time and energy
token_time_energy = {}
for i in range(len(inference_token_nums)):
    if inference_token_nums[i] in token_time_energy:
        token_time_energy[inference_token_nums[i]]["time"].append(inference_times[i])
        token_time_energy[inference_token_nums[i]]["energy"].append(inference_energies[i])
    else:
        token_time_energy[inference_token_nums[i]] = {"time": [inference_times[i]], "energy": [inference_energies[i]]}
for token in token_time_energy:
    token_time_energy[token]["time"] = sum(token_time_energy[token]["time"]) / len(token_time_energy[token]["time"])
    token_time_energy[token]["energy"] = sum(token_time_energy[token]["energy"]) / len(token_time_energy[token]["energy"])

# plot
fig, ax1 = plt.subplots()
ax1.set_xlabel('Generation Length (Tokens)')
ax1.set_ylabel('Time (s)', color='tab:blue')
ax1.plot(list(token_time_energy.keys()), [token_time_energy[token]["time"] for token in token_time_energy], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Energy (J)', color='tab:red')
ax2.plot(list(token_time_energy.keys()), [token_time_energy[token]["energy"] for token in token_time_energy], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
fig.tight_layout()
# save
plt.savefig("qwen_7b_chat_time_energy_1k.png")

