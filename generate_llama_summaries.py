import transformers
from transformers import pipeline
import torch
from huggingface_hub import login
import os

# Login to Hugging Face
login(token="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")

# Check PyTorch and CUDA availability
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Function to load the model and tokenizer
def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, truncation=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


# Function to set up the pipeline
def setup_pipeline(model, tokenizer):
    # Set `pad_token_id` to `eos_token_id` to avoid errors in open-end generation
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, pad_token_id=tokenizer.eos_token_id)
    return pipe


# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model, tokenizer = load_model(model_name)
pipe = setup_pipeline(model, tokenizer)


# Function to generate text based on prompts
def generate_text(prompt, pipe, max_new_tokens=100):
    response = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )
    return response[0]['generated_text']


# Define constants for the prompt
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = f"""{INTRO_BLURB}
{INSTRUCTION_KEY}
{{instruction}}
{RESPONSE_KEY}
"""


# Function to read the prompt from a file with UTF-8 encoding
def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read().strip()
    return prompt

# Function to save the generated text to a file
def save_generated_text(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


# Define the directories
summaries_dir = "summaries"
references_dir = "generated_summaries"

os.makedirs(references_dir, exist_ok=True)

for filename in os.listdir(summaries_dir):
    if filename.endswith(".txt"):
        # Read the prompt from the file
        prompt_file_path = os.path.join(summaries_dir, filename)
        instruction = read_prompt_from_file(prompt_file_path)

        # Format the prompt according to the specified format
        formatted_prompt = PROMPT_FOR_GENERATION_FORMAT.format(
            instruction=instruction)

        # Generate the text
        output = generate_text(formatted_prompt, pipe, max_new_tokens=600)

        # Define the output file path
        output_file_path = os.path.join(references_dir, filename)

        # Save the generated text to the references directory
        save_generated_text(output_file_path, output)

        print(
            f"Processed {filename} and saved the generated summary to {output_file_path}")

print("All files have been processed and summaries have been generated.")
