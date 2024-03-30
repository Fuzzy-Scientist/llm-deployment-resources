# pip install transformers==4.38.1
from transformers import pipeline
from .utils import track_time

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompts = [prompt] * 5

with track_time(prompts):
    outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

print(outputs[0][0]["generated_text"])

## cpu
## latency: 48 s

## gpu
## latency: 2.9073479175567627s
## throughput: 0.34 inputs/s
