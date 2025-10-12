import argparse
import json
import os
import sys
from typing import Any, Dict, List


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Update or create a versions.json file with a new version."
    )
    parser.add_argument(
        "new_version",
        type=str,
        help="The new version string to add (e.g., 'v2.1.0').",
    )
    parser.add_argument(
        "versions_file",
        type=str,
        help="Path to the versions.json file.",
    )
    return parser.parse_args()


def read_versions_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads and parses versions data from a JSON file.

    If the file does not exist, it informs the user and returns an empty list,
    allowing a new file to be created.

    Args:
        file_path (str): The path to the versions.json file.

    Returns:
        List[Dict[str, Any]]: The parsed list of version dictionaries, or an
        empty list if the file doesn't exist.

    Raises:
        json.JSONDecodeError: If the file exists but contains invalid JSON.
    """
    print(f"Attempting to read data from {file_path}...")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"File not found. A new '{os.path.basename(file_path)}' "
            f"will be created."
        )
        return []  # Return an empty list to start fresh
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {file_path}. "
            f"Please check its format.",
            file=sys.stderr,
        )
        sys.exit(1)


def update_versions_list(
    versions_data: List[Dict[str, Any]], new_version: str
) -> List[Dict[str, Any]]:
    """
    Adds a new version to the list, marking it as stable and others as not.

    This function performs the following steps:
    1. Checks if the new version already exists to prevent duplicates.
    2. Sets the 'stable' flag to False for all existing versions.
    3. Creates a new dictionary for the new version, marked as 'stable: True'.
    4. Prepends the new version to the list.

    Args:
        versions_data (List[Dict[str, Any]]): The current list of versions.
        new_version (str): The new version string to add.

    Returns:
        List[Dict[str, Any]]: The updated list of versions.
    """
    print(f"Processing new version '{new_version}'...")

    # Check for duplicates
    if any(v.get("version") == new_version for v in versions_data):
        print(
            f"Warning: Version '{new_version}' already exists. "
            f"No changes made."
        )
        return versions_data

    # Mark all existing versions as not stable
    updated_old_versions = [{**v, "stable": False} for v in versions_data]

    # Create the new version entry
    new_version_entry = {"version": new_version, "stable": True}

    # Prepend the new version to the list
    return [new_version_entry] + updated_old_versions


def write_versions_data(file_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Writes the versions data back to a JSON file with pretty-printing.

    This function will create the file and any necessary parent directories.

    Args:
        file_path (str): The path to the versions.json file.
        data (List[Dict[str, Any]]): The data to write.
    """
    print(f"Writing updated data to {file_path}...")
    try:
        # Ensure the parent directory exists before trying to write the file.
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")  # Add a trailing newline for POSIX compliance
    except IOError as e:
        print(
            f"Error: Could not write to file {file_path}: {e}", file=sys.stderr
        )
        sys.exit(1)


def main():
    """
    Main function to orchestrate the update process.
    """
    args = parse_arguments()

    current_versions = read_versions_data(args.versions_file)

    updated_versions = update_versions_list(current_versions, args.new_version)

    # Only write if there was a change. This is always true for new files.
    if current_versions != updated_versions:
        write_versions_data(args.versions_file, updated_versions)
        print(
            f"\n✅ Successfully updated "
            f"{os.path.basename(args.versions_file)} with version "
            f"'{args.new_version}'"
        )
    else:
        print("\nNo changes were necessary.")


if __name__ == "__main__":
    main()
