from zeus.monitor import ZeusMonitor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from abc import ABC, abstractmethod

class test(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def do_something(self):
        pass

class qwen_test(test):
    """
    This is a test class designed for Qwen-7B-Chat in huggingface.
    """
    def __init__(self) -> None:
        super().__init__()

    def do_something(self):
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], log_file="qwen_1k.log", approx_instant_energy=True)
        '''
        参数解释
        gpu_indices: 监测所有CUDA设备的索引。这些GPU的时间/能耗测量将同时开始和结束（即同步）。如果为None，则将使用所有可用的GPU。如果设置了CUDA_VISIBLE_DEVICES，将尊重其设置，例如，当CUDA_VISIBLE_DEVICES=2,3时，传递到gpu_indices的GPU索引1将被解释为CUDA设备3。支持使用逗号分隔的索引格式的CUDA_VISIBLE_DEVICES。

        approx_instant_energy: 当测量窗口的执行时间短于NVML能耗计数器的更新周期时，可能观察到能耗为零的情况。在这种情况下，如果approx_instant_energy为True，则窗口的能耗将通过将当前瞬时功耗乘以窗口的执行时间来进行估算。这应该比零更好地估计，但仍然是一种近似值。

        log_file: 日志CSV文件的路径。如果为None，则禁用日志记录。
        '''

        # Note: The default behavior now has injection attack prevention off.
        monitor.begin_window("init_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
        init_tokenizer_eres = monitor.end_window("init_tokenizer")

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use auto mode, automatically select precision based on the device.
        monitor.begin_window("init_model")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
        init_model_eres = monitor.end_window("init_model")

        # tell the model to generate responses of diferent lengthes from 1 to 100 (words)
        # run this for 50 times, collect the time and energy consumption for each generation, get data (tie, energy, generation length in tokens)
        # plot two figures, one for time, one for energy, with x-axis as generation length in tokens
        # begin
        inference_token_nums = []
        inference_times = []
        inference_energies = []
        for i in range(5000):
            monitor.begin_window(f"generate_{i}")
            response, history = model.chat(tokenizer, "please generate a sentence with 0-1000 words", history=None)
            eres = monitor.end_window(f"generate_{i}")
            inference_token_nums.append(len(tokenizer.encode(response)))    
            # Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
            inference_times.append(eres.time)
            inference_energies.append(eres.total_energy)
        # end
        # save the data (including the model, tokenizer, and the data) to a csv file
        with open("inference_data.csv", "w") as f:
            for i in range(len(inference_token_nums)):
                f.write(f"{inference_token_nums[i]},{inference_times[i]},{inference_energies[i]}\n")
        

        return init_tokenizer_eres, init_model_eres, inference_token_nums, inference_times, inference_energies
