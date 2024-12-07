import time
import tracemalloc

# records time and space while running the given function
# returns:
#  time taken in milliseconds
#  the final amount of memory allocated after the function has finished
#  the most amount of memory allocated at any point while the function runs
#  anything returned by the function
def statrun(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time_ns()
    funcreturn = func(*args, **kwargs)
    end_time = time.time_ns()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_taken = (end_time - start_time) / 1000000
    return time_taken, current, peak, funcreturn
