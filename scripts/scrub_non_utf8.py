
import sys
import os

def scrub_file(filepath):
    """
    Reads a file, removes non-UTF-8 characters, and writes it back.
    """
    try:
        if not os.path.isfile(filepath):
            print(f"Skipping directory: {filepath}")
            return

        # Read as bytes
        with open(filepath, 'rb') as f:
            original_bytes = f.read()

        # Decode to string, ignoring errors
        cleaned_string = original_bytes.decode('utf-8', 'ignore')

        # Encode back to bytes
        cleaned_bytes = cleaned_string.encode('utf-8')

        if original_bytes != cleaned_bytes:
            print(f"Scrubbing non-UTF-8 characters from {filepath}")
            with open(filepath, 'wb') as f:
                f.write(cleaned_bytes)
        else:
            print(f"No changes needed for {filepath}")


    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrub_non_utf8.py <file_or_directory_path> ...", file=sys.stderr)
        sys.exit(1)

    for path_arg in sys.argv[1:]:
        if os.path.isdir(path_arg):
            for dirpath, _, filenames in os.walk(path_arg):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    scrub_file(filepath)
        elif os.path.isfile(path_arg):
            scrub_file(path_arg)
        else:
            print(f"Path not found: {path_arg}", file=sys.stderr)
