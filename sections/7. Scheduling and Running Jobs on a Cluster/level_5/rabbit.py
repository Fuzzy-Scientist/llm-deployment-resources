# pip install pika
import pika

RABBIT_USER = "rabbitmq_user"
RABBIT_PASS = "pass"

HEAD_IP = "your_head_node_ip"
RABBIT_PORT = 5672


class RabbitBuffer:
    def __init__(self, queue_name: str) -> None:
        self.queue_name = queue_name

        self.credentials = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(HEAD_IP, RABBIT_PORT, "/", self.credentials))

        self.channel = self.connection.channel()
        self.queue = self.channel.queue_declare(queue=self.queue_name, durable=True)

    def produce(self, messages: list[str]):
        for message in messages:
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=message,
                properties=pika.BasicProperties(delivery_mode=2),  # make messages persistent
            )

    def consume(self, num_messages: int):
        messages = []
        for _ in range(num_messages):
            method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_name)
            if method_frame:
                messages.append(body)
                self.channel.basic_ack(method_frame.delivery_tag)
        return messages
