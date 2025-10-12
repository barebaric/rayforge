import argparse
import json
import os
import shutil
import sys
import yaml
from typing import TypeVar, overload


def env_constructor(loader, node):
    value = loader.construct_scalar(node)
    return os.getenv(value, None)


def unknown_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


yaml.SafeLoader.add_constructor("!ENV", env_constructor)
yaml.SafeLoader.add_multi_constructor("", unknown_constructor)


def parse_arguments():
    """Parses command-line arguments for the entire build prep process."""
    parser = argparse.ArgumentParser(
        description="Prepares a temporary source directory for the MkDocs "
        "build and generates a deployment-specific configuration."
    )
    # Arguments for file preparation
    parser.add_argument(
        "source_dir",
        help="Path to the source website directory (e.g., 'website/').",
    )
    parser.add_argument(
        "tmp_dir",
        help="Path to the temporary directory to prepare for the build.",
    )

    # Arguments for config generation
    parser.add_argument(
        "base_config_name",
        help="Filename of the base mkdocs.yml inside the source.",
    )
    parser.add_argument(
        "versions_json", help="Path to the versions.json file to read from."
    )
    parser.add_argument(
        "current_version",
        help="The version of the docs being built (e.g., '0.22').",
    )
    parser.add_argument(
        "output_config_name",
        help="Filename for the generated temporary config file.",
    )

    return parser.parse_args()


def update_nav_paths(nav_item, version):
    """Recursively prepends the version to paths starting with 'docs/'."""
    if isinstance(nav_item, dict):
        return {
            key: update_nav_paths(value, version)
            for key, value in nav_item.items()
        }
    if isinstance(nav_item, list):
        return [update_nav_paths(item, version) for item in nav_item]
    if isinstance(nav_item, str) and nav_item.startswith("docs/"):
        return f"docs/{version}/{nav_item[len('docs/') :]}"
    return nav_item


T = TypeVar("T")


@overload
def transform_config_paths(
    item: dict, dev_prefix="docs/", deploy_prefix="docs/latest/"
) -> dict: ...


@overload
def transform_config_paths(
    item: list, dev_prefix="docs/", deploy_prefix="docs/latest/"
) -> list: ...


@overload
def transform_config_paths(
    item: T, dev_prefix="docs/", deploy_prefix="docs/latest/"
) -> T: ...


def transform_config_paths(
    item, dev_prefix="docs/", deploy_prefix="docs/latest/"
):
    """
    Recursively finds string values in a config structure (dict or list)
    and replaces a development path prefix with a deployment one.
    """
    if isinstance(item, dict):
        return {
            key: transform_config_paths(value, dev_prefix, deploy_prefix)
            for key, value in item.items()
        }
    if isinstance(item, list):
        return [
            transform_config_paths(sub_item, dev_prefix, deploy_prefix)
            for sub_item in item
        ]
    if isinstance(item, str) and item.startswith(dev_prefix):
        return deploy_prefix + item[len(dev_prefix) :]
    return item


def create_recursive_symlinks(source_root, dest_root):
    """
    Walks the source_root and creates a parallel tree of symlinks in dest_root,
    mirroring the exact structure of the source.
    """
    for dirpath, dirnames, filenames in os.walk(source_root):
        # Determine the corresponding path in the destination tree
        relative_path = os.path.relpath(dirpath, source_root)
        dest_dir = os.path.join(dest_root, relative_path)

        # Create the mirrored directories
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Create symlinks for all files in the current directory
        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            dest_link = os.path.join(dest_dir, filename)
            # Calculate the link target relative to the link's location
            link_target = os.path.relpath(source_file, dest_dir)
            os.symlink(link_target, dest_link)


def main():
    """Main function to orchestrate the build preparation."""
    args = parse_arguments()

    # Part 1: Prepare the temporary source directory
    print(f"Preparing temporary source directory at {args.tmp_dir}...")
    if os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    shutil.copytree(args.source_dir, args.tmp_dir, symlinks=True)

    content_dir = os.path.join(args.tmp_dir, "content")
    docs_content_path = os.path.join(content_dir, "docs")

    if os.path.isdir(docs_content_path):
        versioned_docs_path = os.path.join(
            docs_content_path, args.current_version
        )

        temp_renamed_path = os.path.join(
            content_dir, "docs_renamed_for_versioning"
        )
        shutil.move(docs_content_path, temp_renamed_path)
        os.makedirs(docs_content_path)
        shutil.move(temp_renamed_path, versioned_docs_path)
        print(
            f"Moving documentation content to versioned directory: "
            f"{versioned_docs_path}"
        )

        # Create the RECURSIVE symlink structure
        create_recursive_symlinks(
            versioned_docs_path, os.path.join(docs_content_path, "latest")
        )

    else:
        print(
            "Warning: No 'docs' directory found in source. "
            "Continuing without versioned docs."
        )

    # Part 2: Generate the temporary deploy config
    print("Generating temporary mkdocs config...")
    base_config_path = os.path.join(args.tmp_dir, args.base_config_name)
    output_config_path = os.path.join(args.tmp_dir, args.output_config_name)

    try:
        with open(args.versions_json, "r") as f:
            versions_data = json.load(f)
    except FileNotFoundError:
        print(
            f"Info: {args.versions_json} not found. Building without "
            f"version switcher."
        )
        versions_data = []

    switcher_versions = []
    if versions_data:
        for item in versions_data:
            version_str = item.get("version")
            if version_str:
                switcher_versions.append(
                    {"version": version_str, "title": version_str}
                )
        if switcher_versions:
            switcher_versions[0]["aliases"] = ["stable", "latest"]

    try:
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)
    except (IOError, yaml.YAMLError) as e:
        sys.exit(f"Error reading base config '{base_config_path}': {e}")

    versioned_nav = update_nav_paths(
        base_config.get("nav", []), args.current_version
    )

    base_extra_config = base_config.get("extra", {})
    deploy_extra = transform_config_paths(base_extra_config)

    deploy_extra["version"] = args.current_version
    deploy_extra["latest_docs"] = "docs/latest"

    deploy_config = {
        "INHERIT": args.base_config_name,
        "docs_dir": "content",
        "extra": deploy_extra,
        "nav": versioned_nav,
    }

    if switcher_versions:
        deploy_config["extra"]["versions"] = switcher_versions
        print(
            f"Successfully prepared version switcher for "
            f"{len(switcher_versions)} versions."
        )

    try:
        with open(output_config_path, "w") as f:
            yaml.dump(deploy_config, f, sort_keys=False)
        print(f"Successfully generated deploy config at {output_config_path}")

    except IOError as e:
        print(f"Error writing temporary config: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
