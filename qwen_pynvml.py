from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import pynvml
import atexit
import time
import threading
import matplotlib.pyplot as plt


def _infer_energy_update_period_single(nvml_handle: pynvml.c_nvmlDevice_t) -> float:
    """Infer the update period of the NVML energy acquire for a single GPU."""
    # Collect 1000 samples of energy with timestamps.
    time_energy_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(1000)]
    for i in range(len(time_energy_samples)):
        time_energy_samples[i] = (
            time.time(),
            pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle),
        )

    # Find the timestamps when the energy readings changed.
    changed_times = []
    prev_energy = time_energy_samples[0][1]
    for t, p in time_energy_samples:
        if p != prev_energy:
            changed_times.append(t)
            prev_energy = p

    # Compute the minimum time difference between power change timestamps.
    return min(time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:]))


def infer_energy_update_period(nvml_handles: list[pynvml.c_nvmlDevice_t]) -> float:
    """Infer the update period of the NVML enrergy acquire for a list of GPUs.

    NVML counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    pynvml.nvmlInit()

    # For each unique GPU model, infer the update period.
    update_period = 10
    gpu_models_covered = set()
    for handle in nvml_handles:
        if (model := pynvml.nvmlDeviceGetName(handle)) not in gpu_models_covered:
            print(
                "GPU: " + str(model) +  ", inferring NVML energy acquire update period."
            )
            gpu_models_covered.add(model)
            detected_period = _infer_energy_update_period_single(handle)
            print(
                "Energy acquire update period for %s is %.2f s"% (model, detected_period)
            )
            if update_period > detected_period:
                update_period = detected_period

    pynvml.nvmlShutdown()

    # # Target half the update period to ensure that the energy fetch is fast enough.
    # update_period /= 2.0

    # # Anything less than ten times a second is probably too slow.
    # if update_period > 0.1:
    #     print(
    #         "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
    #         update_period,
    #     )
    #     update_period = 0.1
    return update_period


# Function to read GPU total energy.
def read_gpu_energy():
    power_info = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return power_info / 1000  # Convert to joules.

# Initialize NVML.
pynvml.nvmlInit()
atexit.register(pynvml.nvmlShutdown)
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you have only one GPU.
update_period = infer_energy_update_period([handle])   # Infer the energy update period.
print("Inferred energy update period: %.2f s" % update_period)

device = torch.device("cuda")
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()


running = True  # Flag variable to control the while loop.
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
file_name = f"log_{current_time}.log"
# Function to continuously read GPU energy and log it.
def read_energy_background():
    log_file = open(file_name, "w")
    while running:
        gpu_energy = read_gpu_energy()
        timestamp = str(int(time.time() * 1000000))
        log_entry = f'"ts": {timestamp}, "energy": {gpu_energy}\n'
        log_file.write(log_entry)
        print(log_entry)
        time.sleep(update_period)
    log_file.close()

# Start the background thread to read GPU energy.
energy_thread = threading.Thread(target=read_energy_background)
energy_thread.start()
model = model.to(device)
time.sleep(5)

for i in range(3):
    response, history = model.chat(tokenizer, "please generate some content about some scientist with at least 1000 words", history=None)
    print(response)
    print('------------------------' + str(i) + '------------------------')
    time.sleep(5)
# Stop reading GPU energy after the chat is over.
running = False
energy_thread.join()


with open(file_name, "r") as file:
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
plt.savefig("plot" + current_time + ".png")
