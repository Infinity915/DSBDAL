#!/usr/bin/env python3
import sys

current_word = None
current_count = 0

# Read the sorted output from the mapper
for line in sys.stdin:
    word, count = line.strip().split('\t', 1)
    count = int(count)
    
    if current_word == word:
        current_count += count # Accumulate count for identical words
    else:
        if current_word:
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

# Print the final word
if current_word == word:
    print(f"{current_word}\t{current_count}")