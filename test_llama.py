from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLaMA 3.1 model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Change this to the appropriate model name you downloaded
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a prompt for testing
prompt = "Once upon a time, in a faraway land"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output_ids = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text: ", generated_text)

