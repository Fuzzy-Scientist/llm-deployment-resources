from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm

model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()


def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=64)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated[: generated.find(".") + 1]


generate_text("What's the best way to cook chiken breast?")


def batch_generate_texts(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)["input_ids"]
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated


batch_generate_texts(dataset["instruction"][:1].tolist())
batch_generate_texts(dataset["instruction"][:20].tolist())
batch_generate_texts(dataset["instruction"][:100].tolist())
batch_generate_texts(dataset["instruction"][:200].tolist())
# batch_generate_texts(dataset["instruction"].sample(200).tolist()) # this might crash


# Dynamic batching


def batch_generate_tokens(tokens):
    outputs = model.generate(torch.stack(tokens), max_length=64, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def dynamic_batching(prompts, max_tokens, is_pretokenized=False):
    if not is_pretokenized:
        tokenized_texts = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(model.device)
    else:
        tokenized_texts = prompts

    current_batch = []
    current_batch_size = 0

    for tokenized_text in tokenized_texts:
        if current_batch_size + len(tokenized_text) > max_tokens and current_batch:
            yield batch_generate_tokens(current_batch)

            current_batch, current_batch_size = [], 0

        current_batch.append(tokenized_text)
        current_batch_size += len(tokenized_text)

    # Process final batch
    if current_batch:
        yield batch_generate_tokens(current_batch)
        pass


generator = dynamic_batching(dataset["instruction"][:40].tolist() * 1000, 3200)


from contextlib import contextmanager
import time


@contextmanager
def track_time():
    start = time.time()  # Record start time
    yield
    end = time.time()  # Record end time
    print(f"Execution time: {end - start} seconds")


with track_time():
    for batch_predictions in tqdm(generator):
        continue


def sort_batches(prompts, max_tokens):
    tokenized_texts = tokenizer(prompts, padding=False)["input_ids"]
    sorted_tokens = sorted(tokenized_texts, key=len)

    sorted_batches = {}
    for sorted_token in sorted_tokens:
        length = len(sorted_token)
        if length not in sorted_batches:
            sorted_batches[length] = []

        sorted_batches[length].append(sorted_token)

    for length, sorted_batch in sorted_batches.items():
        tensor_batch = torch.stack([torch.tensor(sorted_token) for sorted_token in sorted_batch]).to(model.device)
        for batch_prediction in dynamic_batching(tensor_batch, max_tokens=max_tokens, is_pretokenized=True):
            yield batch_prediction


generator = sort_batches(dataset["instruction"][:40].tolist() * 1000, 3200)

with track_time():
    for batch_predictions in tqdm(generator):
        print(len(batch_predictions))
