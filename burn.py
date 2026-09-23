import time
print("burning CPU - press Ctrl+C to stop")
start = time.time()
try:
    while True:
        x = 0
        for i in range(10_000_000):
            x += i
        print(f"  running {int(time.time() - start)}s")
except KeyboardInterrupt:
    print("stopped")