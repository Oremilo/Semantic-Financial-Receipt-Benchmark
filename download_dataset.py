from datasets import load_dataset

print("Downloading CORD dataset...")

dataset = load_dataset("naver-clova-ix/cord-v1")

print(dataset)

print("\nTrain Sample:")
print(dataset["train"][0])