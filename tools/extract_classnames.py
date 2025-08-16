import os
import glob
from tqdm import tqdm

label_dir = "/path/to/labelTxt"
output_dir = "/path/to/classnames"

os.makedirs(output_dir, exist_ok=True)
label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
print(f"Found {len(label_files)} label files.")

for label_file in tqdm(label_files, desc="Processing label files"):
    # Read label file
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Extract all class names
    classnames = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 10:  # Ensure correct line format
            classname = parts[8]  # Class name is at position 9 (index 8)
            classnames.append(classname)
    unique_classnames = list(set(classnames))
    output_filename = os.path.basename(label_file)
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(unique_classnames))

print(f"Class name extraction completed, results saved to: {output_dir}")
