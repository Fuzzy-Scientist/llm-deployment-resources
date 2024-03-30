# !pip install ctranslate2
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
from ctranslate2.converters import TransformersConverter
from ctranslate2 import Generator

from contextlib import contextmanager
import time


@contextmanager
def track_time():
    start = time.time()  # Record start time
    yield
    end = time.time()  # Record end time
    print(f"Execution time: {end - start} seconds")


model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Convert the model to CTranslate2
model.save_pretrained("models/gpt-instruct")
tokenizer.save_pretrained("models/gpt-instruct")

converter = TransformersConverter("models/gpt-instruct")
out_path = converter.convert(output_dir="models/gpt-instruct-quant", quantization="float16")

generator = Generator("models/gpt-instruct-quant", device="cuda")

# Dataset
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

prompts = dataset["instruction"].sample(3000, random_state=42).tolist()


# Normal batching
def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def batch_generate_tokens(tokens):
    outputs = model.generate(tokens, max_length=256, pad_token_id=tokenizer.eos_token_id, num_beams=2, repetition_penalty=1.5)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def predict_batch(prompts, batch_size):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]

    for batch in chunker(inputs, batch_size):
        yield batch_generate_tokens(batch.to(model.device))


with track_time():
    for batch_prediction in tqdm(predict_batch(prompts, 32)):
        continue

# Execution time: 242.11289978027344 seconds


# CTranslate2 batching with quantized model
def batch_generate_ctrans(prompts, batch_size):
    inputs = [tokenizer.tokenize(prompt, truncation=True, max_length=128) for prompt in prompts]

    results = generator.generate_batch(inputs, max_length=256, max_batch_size=batch_size, beam_size=2, repetition_penalty=1.5)

    result_ids = [res.sequences_ids[0] for res in results]
    return tokenizer.batch_decode(result_ids, skip_special_tokens=True)


del model
torch.cuda.empty_cache()
with track_time():
    batch_generate_ctrans(prompts, 32)

# Execution time: 150.97192573547363 seconds
