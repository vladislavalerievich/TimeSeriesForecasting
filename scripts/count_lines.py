import os


def count_lines_in_file(filepath):
    """Count the number of lines in a single file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_lines_in_directory(directory, extensions=(".py",)):
    """Recursively count lines in all files with given extensions in a directory"""
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                filepath = os.path.join(root, file)
                try:
                    lines = count_lines_in_file(filepath)
                    total_lines += lines
                    print(f"{filepath}: {lines} lines")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return total_lines


if __name__ == "__main__":
    src_directory = "src"
    if not os.path.exists(src_directory):
        print(f"Error: Directory '{src_directory}' not found.")
        exit(1)

    print(f"Counting lines in Python files under '{src_directory}'...")
    total_lines = count_lines_in_directory(src_directory)
    print(f"\nTotal lines of code: {total_lines}")
