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

    This code just calls huggingface's api to init the tokenizer and model, and then uses zeus to measure corresponding time and energy

    No detail about model inference
    """
    def __init__(self) -> None:
        super().__init__()

    def do_something(self, log_file:str):
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()], log_file=log_file, approx_instant_energy=True)

        # Note: The default behavior now has injection attack prevention off.
        monitor.begin_window("init_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
        init_tokenizer_eres = monitor.end_window("init_tokenizer")

        # define and setup model
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

        with open("inference_data.csv", "w") as f:
            for i in range(len(inference_token_nums)):
                f.write(f"{inference_token_nums[i]},{inference_times[i]},{inference_energies[i]}\n")
        
        return init_tokenizer_eres, init_model_eres, inference_token_nums, inference_times, inference_energies

class nano_gpt(test):
    """"
        This 
    """