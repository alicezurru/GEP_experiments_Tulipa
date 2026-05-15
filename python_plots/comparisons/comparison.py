from pathlib import Path
import difflib

# Define your directory
SCRIPT_DIR = Path(__file__).parent  # or set manually

file1 = SCRIPT_DIR / "model_new_input_4rps.lp"
file2 = SCRIPT_DIR / "model_used_input_4rps.lp"

# --- Function: normalize content (ignore whitespace differences)
def normalize(content):
    return "\n".join(line.strip() for line in content.splitlines() if line.strip())

# --- Compare files
with open(file1, "r") as f1, open(file2, "r") as f2:
    content1 = normalize(f1.read())
    content2 = normalize(f2.read())

if content1 == content2:
    print("✅ The .lp files are logically the same.")
else:
    print("❌ The .lp files are different.\n")

    # Show differences
    with open(file1) as f1, open(file2) as f2:
        diff = difflib.unified_diff(
            f1.readlines(),
            f2.readlines(),
            fromfile=str(file1),
            tofile=str(file2)
        )

        print("🔍 Differences:")
        for line in diff:
            print(line, end="")
