# pip install -U "ray[default]"
import ray
from rabbit import RabbitBuffer

ray.init(address="auto")
from vllm import LLM, SamplingParams


@ray.remote
def predict_batch():
    buffer = RabbitBuffer("llama-queue")

    messages = buffer.consume(5000)
    prompts = [m.decode() for m in messages]

    sampling = SamplingParams(max_tokens=256, seed=42, temperature=0)
    llm = LLM(model="/root/ml-deployment/models/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling)

    results = [output.outputs[0].text for output in outputs]

    result_buffer = RabbitBuffer("llama-results")
    result_buffer.produce(results)

    return results


if __name__ == "__main__":
    future = predict_batch.options(num_gpus=1, num_cpus=1).remote()
    ray.get(future)
    ray.shutdown()

# sumbit command
# ray job submit --submission-id llamma-batch1 --working-dir src/level_five/ -- python ray_batch_job.py

# throughput: 111 inputs/s
