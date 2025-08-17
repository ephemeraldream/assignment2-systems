import torch


import time

print("=== Test 1: float32 + float32 ===")
start_time = time.time()
s = torch.tensor(0,dtype=torch.float32)
for i in range(100000):
    s += torch.tensor(0.01,dtype=torch.float32)
end_time = time.time()
print(f"Result: {s}")
print(f"Time: {(end_time - start_time)*1000:.3f} ms")

print("\n=== Test 2: float16 + float16 ===")
start_time = time.time()
s = torch.tensor(0,dtype=torch.float16)
for i in range(100000):
    s += torch.tensor(0.01,dtype=torch.float16)
end_time = time.time()
print(f"Result: {s}")
print(f"Time: {(end_time - start_time)*1000:.3f} ms")

print("\n=== Test 3: float32 + float16 (implicit conversion) ===")
start_time = time.time()
s = torch.tensor(0,dtype=torch.float32)
for i in range(100000):
    s += torch.tensor(0.01,dtype=torch.float16)
end_time = time.time()
print(f"Result: {s}")
print(f"Time: {(end_time - start_time)*1000:.3f} ms")

print("\n=== Test 4: float32 + float16 (explicit conversion) ===")
start_time = time.time()
s = torch.tensor(0,dtype=torch.float32)
for i in range(100000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
end_time = time.time()
print(f"Result: {s}")
print(f"Time: {(end_time - start_time)*1000:.3f} ms")