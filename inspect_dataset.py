from datasets import load_dataset
import json

dataset = load_dataset("naver-clova-ix/cord-v1")

sample = dataset["train"][0]

print("\n==============================")
print("DATASET KEYS")
print("==============================")

print(sample.keys())

print("\n==============================")
print("FULL SAMPLE")
print("==============================")

for key, value in sample.items():

    print(f"\nKEY: {key}")
    print("TYPE:", type(value))

    if isinstance(value, str):
        print(value[:1000])

    else:
        print(value)