import re
from pathlib import Path

import requests


def check_urls(file_path):
    # If extension is md, check for URLs in markdown files
    if file_path.suffix == ".md":
        url_pattern = re.compile(
            r"\[.*?\]\(https?://[^\s\"\'<>]+(?:[^\s\"\'<>.,;:!?)]|/)+\)\s*\(https?://[^\s\"\'<>]+(?:[^\s\"\'<>.,;:!?)]|/)+\)"
        )
    else:
        url_pattern = re.compile(r"https?://\S+")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        urls = url_pattern.findall(content)
        for url in urls:
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code != 200:
                    print(f"URL {url} returned status code {response.status_code}")
            except requests.RequestException:
                pass  # print(f"URL {url} could not be reached: {e}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    print(f"Checking URLs in {project_dir}")
    for file_path in project_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in {".py", ".md", ".toml"}:
            check_urls(file_path)


if __name__ == "__main__":
    main()
