import os

def process_file(filepath, min_lines=1000, remove_fraction=0.98):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        # Skip binary or unreadable files
        return

    n = len(lines)
    if n <= min_lines:
        return

    keep_fraction = (1 - remove_fraction) / 2  # split between start and end
    keep_count = int(n * keep_fraction)

    # Ensure at least 1 line kept on each side
    keep_count = max(1, keep_count)

    new_lines = lines[:keep_count] + lines[-keep_count:]

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    print(f"Processed: {filepath} ({n} → {len(new_lines)} lines)")


def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            process_file(filepath)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        exit(1)

    root_directory = sys.argv[1]
    process_directory(root_directory)