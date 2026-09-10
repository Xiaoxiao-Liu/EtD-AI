import time
import random

try:
    while True:
        print("running...")
        time.sleep(random.randint(1, 5))
except KeyboardInterrupt:
    print("stopped")