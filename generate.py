from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "./final_gpt2_ai_blog_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

prompt = input("Enter your prompt: ")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
)

print("\nGenerated Article:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
