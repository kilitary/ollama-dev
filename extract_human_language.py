#!/usr/bin/env python3
"""
Extracts only human language data from each line of comments.csv and adds it to lines.csv.
Filters out non-language terms like URLs, user IDs, and other non-human content.
"""

import csv
import re


def is_human_language(text):
    """
    Check if the text is human language and not just technical content.
    Returns True if it's human language, False otherwise.
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # If text is empty, it's not human language
    if not text:
        return False

    # Filter out text that is just a URL
    url_pattern = r"^https?://"
    if re.match(url_pattern, text):
        return False

    # Filter out text that is just an ID or number
    if re.match(r"^\d+$", text):
        return False

    # Filter out text that is just technical jargon or single characters
    if len(text) < 3 and not re.search(r"[а-яА-Я]", text):
        return False

    # Filter out text with too many special characters or non-language symbols
    if len(text) > 0:
        # Count special characters
        special_chars = re.findall(r"[^а-яА-Яa-zA-Z0-9\s\-\.,!?;:\'\"\(\)]", text)
        if len(special_chars) > len(text) * 0.3:  # More than 30% special characters
            return False

    # Filter out text that looks like code or technical identifiers
    code_patterns = [
        r"^[A-Z_]+\d*$",  # LIKE_THIS123
        r"^\d+[A-Za-z]+$",  # 123abc
        r"^[A-Za-z]+\d+$",  # abc123
        r"^[0-9═²⌠îï±]+$",  # Technical symbols
    ]

    for pattern in code_patterns:
        if re.match(pattern, text):
            return False

    return True


def clean_text(text):
    """
    Clean the text by removing or replacing non-language elements.
    """
    # Remove user ID mentions like [id12345|Name]
    text = re.sub(r"\[id\d+\|[^]]*]", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Input and output file paths
input_file = "comments.csv"
output_file = "lines.csv"

# Open the input and output files
with (
    open(input_file, mode="r", encoding="utf-8") as infile,
    open(output_file, mode="w", encoding="utf-8", newline="") as outfile,
):
    # Create a CSV reader and writer
    reader = csv.reader(infile, delimiter="|")
    writer = csv.writer(outfile)

    # Write header to output file
    writer.writerow(["human_language"])

    # Process each line in the input file
    for row_num, row in enumerate(reader, 1):
        # The first column contains the human language data
        if (
            row and len(row) > 0
        ):  # Ensure the row is not empty and has at least one column
            human_language_data = row[0]

            # Clean the text
            cleaned_data = clean_text(human_language_data)

            # Check if it's valid human language
            if is_human_language(cleaned_data):
                writer.writerow([cleaned_data])
            # else:
            #     print(f"Filtered out row {row_num}: {cleaned_data}")

print(f"Extraction complete. Human language data has been written to {output_file}.")
