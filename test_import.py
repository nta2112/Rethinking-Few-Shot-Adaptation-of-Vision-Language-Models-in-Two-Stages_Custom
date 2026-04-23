import os
import sys

# Add the current directory to sys.path to import from datasets
sys.path.append(os.getcwd())

try:
    from datasets.tlu import TLU
    print("Successfully imported TLU dataset class")
    
    # Try to initialize it with dummy paths to check for syntax errors in __init__
    # Note: this will likely fail on read_json if paths are wrong, but we check syntax first
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
