from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import pynvml
import time
import matplotlib.pyplot as plt
import multiprocessing

device = torch.device("cuda")
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
model = model.to(device)

current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
log_name = "log_" + current_time + ".log"
log_file = open(log_name, "w")

# Function to continuously read GPU energy and log it.
# Need to initialize a new NVML in this function because it will be run in a separate process.
def read_energy_background():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while True:
            power_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            gpu_energy = power_info / 1000  # Convert to joules.
            timestamp = str(int(time.time() * 1000000))
            log_entry = f'"ts": {timestamp}, "energy": {gpu_energy}\n'
            log_file.write(log_entry)
            time.sleep(0.1)
    except KeyboardInterrupt:
        return
    finally:
        pynvml.nvmlShutdown()

# Start the background process to read GPU energy.
energy_process = multiprocessing.Process(target=read_energy_background)
energy_process.start()
time.sleep(3)
for i in range(2):
    response, history = model.chat(tokenizer, "please generate some content about some scientist with at least 200 words", history=None)
    print(response)
    print('------------------------' + str(i) + '------------------------')
    time.sleep(3)
energy_process.terminate()
energy_process.join()
log_file.close()



with open(log_name, "r") as file:
    lines = file.readlines()
timestamps = []
energies = []
for line in lines:
    parts = line.split(",")
    timestamp = float(parts[0].split(":")[1].strip().replace('"', ''))
    energy = float(parts[1].split(":")[1].strip().replace('"', ''))
    timestamps.append(timestamp)
    energies.append(energy)
timestamps = [t - min(timestamps) for t in timestamps]
timestamps = [t / 1000000 for t in timestamps]
energies = [e - min(energies) for e in energies]

plt.plot(timestamps, energies)
plt.xlabel("Timestamp (seconds)")
plt.ylabel("Energy (joules)")
plt.savefig("plot_" + current_time + ".png")