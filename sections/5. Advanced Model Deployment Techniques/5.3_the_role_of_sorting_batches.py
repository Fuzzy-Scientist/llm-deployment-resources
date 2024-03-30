from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
from contextlib import contextmanager
import time


@contextmanager
def track_time():
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.2f}s")


model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

prompts = dataset["instruction"].sample(4).tolist()
inputs = tokenizer(prompts, padding=True)["input_ids"]


# print('\n\n'.join(tokenizer.batch_decode(inputs)))
print("\n\n".join(tokenizer.batch_decode(inputs)).replace(tokenizer.eos_token, "[PAD]"))


# Normal batching
def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def batch_generate_tokens(tokens):
    outputs = model.generate(tokens, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def predict_batch(prompts, batch_size):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]

    for batch in chunker(inputs, batch_size):
        yield batch_generate_tokens(batch.to(model.device))


prompts = dataset["instruction"].sample(3000).tolist()

with track_time():
    for batch_prediction in tqdm(predict_batch(prompts, 32)):
        print(len(batch_prediction))
# Execution time: 137.19s


# Sorted Batching
def predict_sorted_batches(prompts, max_batch_size):
    inputs = tokenizer(prompts, padding=False, truncation=True, max_length=512)["input_ids"]

    sorted_tokens = sorted(inputs, key=len)
    sorted_batches = {}
    for sorted_input in sorted_tokens:
        if not len(sorted_input):
            continue

        length = len(sorted_input)
        if length not in sorted_batches:
            sorted_batches[length] = []

        sorted_batches[length].append(sorted_input)

    for length, sorted_batch in sorted_batches.items():
        for batch in chunker(sorted_batch, max_batch_size):
            tensor_batch = torch.tensor(batch).to(model.device)
            yield batch_generate_tokens(tensor_batch)


with track_time():
    for batch_prediction in tqdm(predict_sorted_batches(prompts, 32)):
        print(len(batch_prediction))

# Execution time: 72.74s
