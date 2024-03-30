# !pip install vllm==0.3.3
from .utils import track_time
from vllm import LLM, SamplingParams

llm = LLM(model="models/TinyLlama-1.1B-Chat-v1.0")
tokenizer = llm.get_tokenizer()


messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)


sampling = SamplingParams(max_tokens=256, seed=42, temperature=0)

prompts = [prompt] * 1024

with track_time(prompts):
    outputs = llm.generate(prompts, sampling)

results = [output.outputs[0].text for output in outputs]

print(results[1000])

# latency: 0.7040367126464844s
# throughput: 68.22 inputs/s
