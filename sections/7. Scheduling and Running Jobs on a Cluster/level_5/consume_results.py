from src.level_five.rabbit import RabbitBuffer

buffer = RabbitBuffer("llama-results")

results = buffer.consume(10_000)
len(results)

print(results[9000].decode())
