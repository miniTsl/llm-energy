from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
inputs = tokenizer("Nikola Tesla is ", return_tensors="pt") # <s> Hello my name is
print(inputs)   # {'input_ids': tensor([[    1, 12720,  5130, 26455,  2988,  1117, 29473]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
print(model.generation_config)  # only shows the different ones from default config

outputs = model.generate(**inputs, max_new_tokens=40, num_beams=5, early_stopping=True)
print(outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))