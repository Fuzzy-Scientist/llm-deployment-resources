from contextlib import contextmanager
import time

@contextmanager
def track_time(inputs:list=None):
    start = time.time() # Record the start time
    yield # Pass control back to the context block
    duration = time.time() - start # Calculate the duration

    if inputs is None:
        print(f"Execution time: {duration:.2f} seconds")
    else:
        print(f"Took {duration:.2f} seconds to process {len(inputs)} inputs")