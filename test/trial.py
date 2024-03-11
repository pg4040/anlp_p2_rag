from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="../llama-2-13b-chat.Q4_0.gguf", model_type="llama", gpu_layers=50)

print(llm("AI is going to"))

