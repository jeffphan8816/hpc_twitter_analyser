from mpi4py import MPI
import orjson

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# seh (sentiment each hour) use to store SUM OF SENTIMENT each hour per day
seh = {}
# sed (sentiment each day) use to store SUM OF SENTIMENT each day
sed = {}


def process(data):
    for item in data:
        if isinstance(item, dict):
            itemDoc = item.get("doc")
            if isinstance(itemDoc, dict):
                itemData = itemDoc.get("data")
                date, hour = itemData.get("created_at").split("T")
                hour = hour[:2]
                sentiment = itemData.get("sentiment")

                # The sentiment value must be a non-zero floating-point number
                if isinstance(sentiment, float):
                    if date not in seh:
                        seh[date] = {}
                        sed[date] = sentiment
                    else:
                        sed[date] += sentiment
                    if hour not in seh[date]:
                        seh[date][hour] = sentiment
                    else:
                        seh[date][hour] += sentiment
    # print(seh)


def happiestHour(seh):
    happiest = {}
    maxHappiest = 0
    for date in seh:
        for hour in seh[date]:
            if hour > maxHappiest:
                maxHappiest = hour
                happiest = {date: {hour: seh[date][hour]}}
    print(f"Happiest hour: {happiest[date][hour]} with sentiment: {happiest[date][hour]}")


# use to merge two dictionaries
def merge_dicts(x, y):
    merged = {}

    # Iterate through keys in x
    for key in x:
        # Check if the key is also in y
        if key in y:
            # If both values are dictionaries, recursively merge them
            if isinstance(x[key], dict) and isinstance(y[key], dict):
                merged[key] = merge_dicts(x[key], y[key])
            # If both values are floats, sum them
            elif isinstance(x[key], float) and isinstance(y[key], float):
                merged[key] = x[key] + y[key]
            else:
                # If types don't match, just take the value from x
                merged[key] = x[key]
        else:
            # If key is only in x, take its value
            merged[key] = x[key]

    # Add keys from y that are not in x
    for key in y:
        if key not in x:
            merged[key] = y[key]

    return merged


data = None
start_time = None

if rank == 0:
    start_time = MPI.Wtime()
    with open('dummy_data0.json.json', 'r', encoding='utf-8') as f:
        data = orjson.loads(f.read()).get("rows")
        # created_at = data.get('doc', {}).get('data', {}).get('created_at', None)
        # sentiment = data.get('doc', {}).get('data', {}).get('sentiment', None)
        # data = [created_at, sentiment]
        data = [data[i::size] for i in range(size)]
    print(f"Loading time: {MPI.Wtime() - start_time} \n\n")
data = comm.scatter(data, root=0)

# Process the data
process(data)

# Reduce the data and give back to the root process

merged_seh = comm.reduce(seh, op=merge_dicts, root=0)
merged_sed = comm.reduce(sed, op=merge_dicts, root=0)
if rank == 0:
    merged_seh = [merged_seh[i::size] for i in range(size)]
merged_seh = comm.scatter(merged_seh, root=0)

happiestHour(merged_seh)

if rank == 0:
    print(f"Sentiment per hour: {merged_seh}\n\n")
    print(f"Sentiment per day: {merged_sed}\n\n")
    print(f"Calculating time: {MPI.Wtime() - start_time} \n\n")
