#!/usr/bin/env python3

def count_characters(filename):
    """
    Count the number of characters in a text file using UTF-8 encoding.
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        int: Total number of characters in the file
    """
    try:
        # Open the file with UTF-8 encoding
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Count characters
        char_count = len(content)
        
        print(f"File: {filename}")
        print(f"Total characters: {char_count}")
        
        return char_count
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return -1
    except UnicodeDecodeError:
        print(f"Error: Unable to decode '{filename}' with UTF-8 encoding.")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

if __name__ == "__main__":
    import sys
    
    # Check if a filename was provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python character_counter.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    count_characters(filename)