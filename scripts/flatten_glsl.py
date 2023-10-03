import re
import os
import sys

source_paths = []


def flatten_shader(source_path):
    if source_path in source_paths:
        return ""
    # append source path to list of source paths
    source_paths.append(source_path)

    directory = os.path.dirname(source_path)
    with open(source_path, "r") as file:
        source = file.read()

    includes = re.findall(r'#include \"(.*)\"', source)

    for include in includes:
        included_source = flatten_shader(os.path.join(directory, include))
        source = source.replace(f'#include "{include}"', included_source, 1)

    return source


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flatten_shader.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    flattened_shader = flatten_shader(input_file)

    with open(output_file, "w") as file:
        file.write(flattened_shader)
