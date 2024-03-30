from src.level_five.rabbit import RabbitBuffer
from transformers import AutoTokenizer

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]
tokenizer = AutoTokenizer.from_pretrained("models/TinyLlama-1.1B-Chat-v1.0")

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)


buffer = RabbitBuffer("llama-queue")
buffer.produce([prompt] * 100_000)
