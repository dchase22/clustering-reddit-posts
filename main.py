from datasets import load_dataset
from itertools import islice

DATA_ITEMS = 10000

def main():
    ds = load_dataset("wenknow/reddit_dataset_44", split="train", streaming=True)

    distinct_items = set()

    for value in islice(ds["communityName"], DATA_ITEMS):
        distinct_items.add(value)
    
    if len(distinct_items) > 10:
        print(f"Length: {len(distinct_items)}")
        for i in range(0, 10):
            print(distinct_items.pop())

if __name__ == "__main__":
    main()
