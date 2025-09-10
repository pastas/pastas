import re
from pathlib import Path

import tomlkit

PYPROJECT = "pyproject.toml"
README = "README.md"


def get_deps(folder):
    with open(Path.joinpath(folder, PYPROJECT), "r") as f:
        data = tomlkit.parse(f.read())
    return data["project"]["dependencies"]


def update_readme(deps, folder):
    with open(Path.joinpath(folder, README), "r") as f:
        content = f.read()

    # Find the dependencies bullet list block (lines starting with -   ) between ## Dependencies and the next blank line or non-bullet line
    pattern = r"(## Dependencies[^\n]*\n(?:[^\n]*\n)*?)(-   .*\n)+"
    new_deps = "\n".join([f"-   {dep}" for dep in deps]) + "\n"
    # Replace only the bullet list, keep everything else
    new_content = re.sub(pattern, r"\1" + new_deps, content, flags=re.MULTILINE)

    with open(Path.joinpath(folder, README), "w") as f:
        f.write(new_content)

if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent
    deps = get_deps(project_dir)
    update_readme(deps, project_dir)
