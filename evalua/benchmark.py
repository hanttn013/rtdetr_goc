import onnxruntime as ort
import numpy as np
import time

# =========================
# CONFIG
# =========================
MODEL_PATH = "model/model.onnx"
IMG_SIZE = 640
NUM_RUNS = 200
WARMUP = 20

# =========================
# CREATE SESSION
# =========================
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded:", MODEL_PATH)
print("Input name:", input_name)

# =========================
# DUMMY INPUT
# =========================
input_data = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

# =========================
# WARMUP
# =========================
print("Warmup...")
for _ in range(WARMUP):
    _ = session.run(None, {input_name: input_data})

# =========================
# BENCHMARK
# =========================
print("Running benchmark...")

times = []

for _ in range(NUM_RUNS):
    start = time.time()
    _ = session.run(None, {input_name: input_data})
    end = time.time()

    times.append((end - start) * 1000)  # ms

times = np.array(times)

# =========================
# RESULT
# =========================
mean_latency = np.mean(times)
fps = 1000 / mean_latency

print("\n===== CPU BENCHMARK =====")
print(f"Average Latency: {mean_latency:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"P50: {np.percentile(times, 50):.2f} ms")
print(f"P90: {np.percentile(times, 90):.2f} ms")
print(f"P95: {np.percentile(times, 95):.2f} ms")
print(f"Min: {np.min(times):.2f} ms")
print(f"Max: {np.max(times):.2f} ms")