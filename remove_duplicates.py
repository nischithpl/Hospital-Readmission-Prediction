import nbformat
import hashlib

notebook_path = "notebooks/model_training.ipynb"

# Load notebook
nb = nbformat.read(notebook_path, as_version=4)

seen = {}
duplicates = []

for i, cell in enumerate(nb.cells):

    if cell.cell_type != "code":
        continue

    code = cell.source.strip()

    if not code:
        continue

    # Generate hash of code
    code_hash = hashlib.md5(code.encode()).hexdigest()

    if code_hash in seen:
        duplicates.append((i, seen[code_hash]))
    else:
        seen[code_hash] = i

print("Duplicate cells found:")

for duplicate, original in duplicates:
    print(f"Cell {duplicate} duplicates Cell {original}")

print("Total duplicates:", len(duplicates))