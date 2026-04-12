import os
import glob

label_file = "label.txt"
marker_dir = "data/file_markers_detection"

# Read all marker files and merge them into  dict
label_dict = {}

# Traverse all annotated `.txt` files in the directory
for file_path in glob.glob(os.path.join(marker_dir, "*.txt")):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                fname, tag = parts
                label_dict[fname] = tag

# check the output
print(f"A total of {len(label_dict)} sample labels are read from the annotation file.")

# read `label.txt` and add tags
new_lines = []
missing = []

with open(label_file, 'r') as f:
    for line in f:
        fname = line.strip()
        if fname in label_dict:
            tag = label_dict[fname]
            new_lines.append(f"{fname},{tag}\n")
        else:
            new_lines.append(f"{fname},?\n")
            missing.append(fname)

# Write back to label.txt, overwrite directly
with open(label_file, 'w') as f:
    f.writelines(new_lines)

print(f"The `label.txt` file has been updated, containing {len(new_lines)} lines.")
if missing:
    print(f"Number of samples that can't find label: {len(missing)}")
    for m in missing[:10]:
        print(f"  - {m}")
