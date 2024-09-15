from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLaMA 3.1 model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Change to the correct model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print("LLaMA 3.1 is ready for interactive use! Type your prompt or 'exit' to quit.\n")

while True:
    # Get input prompt from the user
    prompt = input("Enter your prompt: ")

    # Exit if the user types 'exit'
    if prompt.lower() == "exit":
        print("Exiting...")
        break

    # Tokenize the input prompt, ensuring attention_mask is created
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    # Generate text from the model, providing attention_mask and using the pad_token_id
    output_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Pass attention mask
        max_length=200,
        num_return_sequences=1
    )

    # Decode the generated output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display the generated response
    print("\nGenerated response: ", generated_text, "\n")

