import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import time

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["I love this movie!", "This was terrible."]
inputs = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="np")

# Triton client setup
client = httpclient.InferenceServerClient(url="localhost:8000")
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)

# Wrap in Triton infer inputs
infer_inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
    httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
]
infer_inputs[0].set_data_from_numpy(input_ids)
infer_inputs[1].set_data_from_numpy(attention_mask)

# Request output
outputs = [httpclient.InferRequestedOutput("logits")]

# Send inference
start = time.time()
response = client.infer(model_name="bert", inputs=infer_inputs, outputs=outputs)
end = time.time()

# Get result
logits = response.as_numpy("logits")
print("Output logits:\n", logits)
print(f"Latency: {(end - start) * 1000:.2f} ms")
