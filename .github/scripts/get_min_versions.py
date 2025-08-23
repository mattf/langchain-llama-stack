import re
import sys
import tomllib

from packaging.version import parse as parse_version

MIN_VERSION_LIBS = ["langchain-core"]


def get_min_version(version: str) -> str:
    # case ^x.x.x
    _match = re.match(r"^\^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    # case >=x.x.x,<y.y.y
    _match = re.match(r"^>=(\d+(?:\.\d+){0,2}),<(\d+(?:\.\d+){0,2})$", version)
    if _match:
        _min = _match.group(1)
        _max = _match.group(2)
        assert parse_version(_min) < parse_version(_max)
        return _min

    # case x.x.x
    _match = re.match(r"^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    raise ValueError(f"Unrecognized version format: {version}")


def get_min_version_from_toml(toml_path: str):
    # Parse the TOML file
    with open(toml_path, "rb") as file:
        toml_data = tomllib.load(file)

    # Get the dependencies from project.dependencies
    dependencies_list = toml_data["project"]["dependencies"]

    # Convert list of dependency strings to a dictionary
    dependencies = {}
    for dep in dependencies_list:
        # Parse dependency strings like "langchain-core>=0.3.47"
        match = re.match(r"^([a-zA-Z0-9_-]+)([><=!~^].*)$", dep)
        if match:
            name = match.group(1)
            version = match.group(2)
            # Remove the operator prefix for our parsing
            if version.startswith(">="):
                version = version[2:]
            elif version.startswith("=="):
                version = version[2:]
            elif version.startswith("^"):
                version = version[1:]
            dependencies[name] = version

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in MIN_VERSION_LIBS:
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            # Get the version string
            version_string = dependencies[lib]

            # Use parse_version to get the minimum supported version from version_string
            min_version = get_min_version(version_string)

            # Store the minimum version in the min_versions dictionary
            min_versions[lib] = min_version

    return min_versions


# Get the TOML file path from the command line argument
toml_file = sys.argv[1]

# Call the function to get the minimum versions
min_versions = get_min_version_from_toml(toml_file)

print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))  # noqa: T201
