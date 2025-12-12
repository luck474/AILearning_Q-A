"""
Shared configuration for the quiz question bank scripts.
"""
import os

# Directory paths (relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(ROOT_DIR, "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "assets", "js")

# Course definitions
COURSES = [
    {
        "id": "deep_learning",
        "name": "深度学习 (Deep Learning)",
        "data_file": "deep_learning.json",
        "raw_file": "题库.txt"
    },
    {
        "id": "javaee",
        "name": "JavaEE (Spring/Boot/MVC)",
        "data_file": "javaee.json",
        "raw_file": "JavaEE题库.txt"
    }
]

def get_data_path(filename: str) -> str:
    """Get absolute path to a data file."""
    return os.path.join(DATA_DIR, filename)

def get_raw_path(filename: str) -> str:
    """Get absolute path to a raw source file."""
    return os.path.join(RAW_DIR, filename)

def get_output_path(filename: str) -> str:
    """Get absolute path to an output file."""
    return os.path.join(OUTPUT_DIR, filename)
