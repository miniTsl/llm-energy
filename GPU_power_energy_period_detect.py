import pynvml
import time
import torch
import atexit
import multiprocessing as mp

"""Measure the update periods of the NVML power counter and energy collection on GPU(s)."""

def infer_counter_update_period(nvml_handles: list[pynvml.c_nvmlDevice_t]) -> float:
    """Infer the update period of the NVML power counter.

    NVML counters can update as slow as 10 Hz depending on the GPU model (normally), so
    there's no need to poll them too faster than that. This function infers the
    update period for 'each' unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.

    But in AIoT machine, we have only one GeForece 3090, so later _infer_counter_update_period_single is enough.
    """
    pynvml.nvmlInit()

    # For each unique GPU model, infer the update period.
    update_period = 0.0
    gpu_models_covered = set()
    for handle in nvml_handles:
        if (model := pynvml.nvmlDeviceGetName(handle)) not in gpu_models_covered:
            print(
                "Detected %s, inferring NVML power counter update period.", model
            )
            gpu_models_covered.add(model)
            detected_period = _infer_counter_update_period_single(handle)
            print(
                "Counter update period for %s is %.2f s",
                model,
                detected_period,
            )
            if update_period > detected_period:
                update_period = detected_period

    pynvml.nvmlShutdown()

    # Target half the update period to ensure that the counter is enough.
    update_period /= 2.0

    # Anything less than ten times a second is probably too slow.
    if update_period > 0.1:
        print(
            "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
            update_period,
        )
        update_period = 0.1
    return update_period

def _infer_counter_update_period_single(nvml_handle: pynvml.c_nvmlDevice_t) -> float:
    """Infer the update period of the NVML power counter for a single GPU."""
    # Collect samples of the power counter with timestamps.
    time_power_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(10000000)]
    for i in range(len(time_power_samples)):
        time_power_samples[i] = (
            time.time(),
            pynvml.nvmlDeviceGetPowerUsage(nvml_handle),
        )

    # Find the timestamps when the power readings change.
    changed_times = []
    prev_power = time_power_samples[0][1]
    for t, p in time_power_samples:
        if p != prev_power:
            changed_times.append(t)
            prev_power = p

    # Compute the minimum time difference between power change timestamps.
    return min(time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:]))


# compute reandom matrix computation on GPU, and keep looping to keep the GPU busy and power consumption high
def loopoing():
    while True:
        x = torch.rand(20000, 20000).to("cuda")
        x = torch.mm(x, x)

# Start the loopoing function in a new process
p = mp.Process(target=loopoing)
p.start()
time.sleep(5)
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
print(_infer_counter_update_period_single(nvml_handle))
pynvml.nvmlShutdown()
time.sleep(1)
p.terminate()
p.join()

"""for 3090, the power counter update period is roughly 0.1 s, which is 10Hz."""




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

    # Target half the update period to ensure that the energy fetch is fast enough.
    update_period /= 2.0

    # Anything less than ten times a second is probably too slow.
    if update_period > 0.1:
        print(
            "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
            update_period,
        )
        update_period = 0.1
    return update_period


# Initialize NVML.
pynvml.nvmlInit()
atexit.register(pynvml.nvmlShutdown)
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you have only one GPU.
update_period = _infer_energy_update_period_single(handle)   # Infer the energy update period.
print("Inferred energy update period: %.2f s" % update_period)

"""for 3090, the energy collection update period is roughly 0.1 s, which is 10Hz."""