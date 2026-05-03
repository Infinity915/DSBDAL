#!/usr/bin/env python3
import sys

# Read input line by line
for line in sys.stdin:
    words = line.strip().split()
    
    # Output each word with a count of 1
    for word in words:
        print(f"{word}\t1")