import os

path = "data/DevanagariHandwrittenCharacterDataset/Test/character_1_ka"

file_count = sum(len(files) for _, _, files in os.walk(path))
print(f"Number of files in '{path}': {file_count}")