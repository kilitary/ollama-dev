import time
import random

iteration = 0
temperature = 0.0
n_threads = 8
num_ctx = 2048
num_batch = 256
iid = time.monotonic_ns()
nbit = random.randrange(0, 64)
outer_engine_random_seed = int(time.time_ns() - int(time.time()) ^ nbit)
random.seed(outer_engine_random_seed)
internal_model_random_seed = int(outer_engine_random_seed ^ random.randrange(0, 64))
selected_model = 'mistral-nemo:latest'  # sola
