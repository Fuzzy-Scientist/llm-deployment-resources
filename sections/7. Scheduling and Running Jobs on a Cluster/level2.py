from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import track_time


model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
input_ids = tokenizer([prompt] * 256, return_tensors="pt").to("cuda")

with track_time(input_ids["input_ids"]):
    outputs = model.generate(**input_ids, max_length=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# latency:  2.6578898429870605s
# throughput:  14.10 inputs/s
