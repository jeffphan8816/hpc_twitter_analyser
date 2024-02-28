from mpi4py import MPI
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('rank = ', rank)
print('size = ', size)


def process(item):
    # Replace this with your actual data processing function
    # Access the "id" values
    if isinstance(item, list):
        # If there are multiple records in an array
        for record in item:
            tweet_id = record.get("id")
            print(f"Tweet ID: {tweet_id} and worker: {rank}")

    elif isinstance(item, dict):
        # If there is a single record as an object
        tweet_id = item.get("id")
        print(f"Tweet ID: {tweet_id} and worker: {rank}")
    return item


data = None
if rank == 0:
    with open('twitter-1mb.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = [data[i::size] for i in range(size)]

data = comm.scatter(data, root=0)

# Process the data
processed_data = [process(item) for item in data]

# Gather the data back to the root process
gathered_data = comm.gather(processed_data, root=0)

