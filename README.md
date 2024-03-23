# llm-energy
test scripts for fine-grained llm energy profiling

## 3.23进展
采用pytorch hook机制对Qwen-chat-7b的不同module进行了能耗测试。

在模型源码中的修改只有一处:`transformers/generation/utils.py`中的GenerationMixin类中的sample方法，加入了对输出token数的控制