import multiprocessing
import math

# def divide_work(work, num_workers):
#     # determine the number of items per worker
#     items_per_worker = math.ceil(len(work) / num_workers)

#     # divide the work into chunks
#     work_chunks = [work[i:i + items_per_worker] for i in range(0, len(work), items_per_worker)]

#     return work_chunks

def divide_work(work, num_workers):
    # New version of divide work, should be correct. 
    # Calculate the number of items per worker, rounding down
    items_per_worker = len(work) // num_workers

    # Calculate the number of workers that will handle one extra item
    extra_items = len(work) % num_workers

    work_chunks = []
    start = 0

    for i in range(num_workers):
        # Determine the end of the current chunk
        end = start + items_per_worker + (1 if i < extra_items else 0)

        # Slice the work chunk and add it to the list
        work_chunks.append(work[start:end])

        # Update the start for the next chunk
        start = end

    return work_chunks


def run_mp(runfun, all_tasks, num_processes, *args): 
    task_pools = divide_work(all_tasks, num_workers=num_processes)
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    for i in range(num_processes):
        pool.apply_async(runfun, args=(task_pools[i], *args))

    # Close the pool to free up resources
    pool.close()
    pool.join()