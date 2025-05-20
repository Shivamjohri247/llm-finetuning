"""
Benchmark pre/post optimization performance (latency, throughput).
"""
import time

def benchmark_inference(fn, inputs, n=10):
    times = []
    for _ in range(n):
        start = time.time()
        fn(*inputs)
        times.append(time.time() - start)
    avg_latency = sum(times) / n
    return avg_latency, 1/avg_latency if avg_latency > 0 else 0
