#!/bin/bash

# Function to run test case
run_test_case() {
    echo "Running test case for module: $1"
    python qwen_chat_7b_zeus_hook.py --hook True --module $1
    sleep 300  # Sleep for 5 minutes to cool down the GPU
    echo "" # Add a new line
}

run_test_case "lm_head"
run_test_case "transformer_ln_f"
run_test_case "transformer_wte"
run_test_case "transformer_drop"
run_test_case "transformer_rotary_emb"
run_test_case "transformer_h_QWenBlock"
run_test_case "transformer_h_QWenBlock_RMSNorm"
run_test_case "transformer_h_QWenBlock_QWenAttention"
run_test_case "transformer_h_QWenBlock_QWenMLP"
python qwen_chat_7b_zeus_hook.py

echo "All test cases completed."