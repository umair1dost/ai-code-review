"""
mine_repositories.py
Mines open-source GitHub repositories for buggy/clean module labels
using the SZZ algorithm on commit messages.

Usage:
    python mine_repositories.py --config ../../configs/mining_config.yaml

Requires: GITHUB_TOKEN environment variable set.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
BUG_PATTERN = re.compile(r"\b(fix|bug|defect|issue|error|patch|fault|repair)\b", re.IGNORECASE)

HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}",
           "Accept": "application/vnd.github.v3+json"}


def search_repositories(language: str, min_stars: int, max_repos: int) -> List[Dict]:
    repos = []
    page = 1
    while len(repos) < max_repos:
        params = {
            "q": f"language:{language} stars:>{min_stars}",
            "sort": "stars", "order": "desc",
            "per_page": 30, "page": page,
        }
        r = requests.get(f"{GITHUB_API}/search/repositories", headers=HEADERS, params=params)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            break
        repos.extend(items)
        page += 1
        time.sleep(1)  # Respect rate limits
    return repos[:max_repos]


def get_bug_fix_commits(owner: str, repo: str) -> List[Dict]:
    commits = []
    page = 1
    while True:
        r = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/commits",
            headers=HEADERS, params={"per_page": 100, "page": page}
        )
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        for c in data:
            msg = c.get("commit", {}).get("message", "")
            if BUG_PATTERN.search(msg):
                commits.append({"sha": c["sha"], "message": msg})
        page += 1
        time.sleep(0.5)
        if len(data) < 100:
            break
    return commits


def get_changed_files(owner: str, repo: str, sha: str) -> List[str]:
    r = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}", headers=HEADERS)
    if r.status_code != 200:
        return []
    return [f["filename"] for f in r.json().get("files", [])]


def mine(config: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for lang in config.get("languages", ["python", "java"]):
        repos = search_repositories(
            language=lang,
            min_stars=config.get("min_stars", 100),
            max_repos=config.get("max_repos_per_language", 256),
        )
        logger.info("Found %d %s repositories to mine", len(repos), lang)

        for repo in repos:
            owner = repo["owner"]["login"]
            name  = repo["name"]
            logger.info("Mining %s/%s ...", owner, name)

            bug_commits = get_bug_fix_commits(owner, name)
            buggy_files = set()
            for commit in bug_commits:
                files = get_changed_files(owner, name, commit["sha"])
                ext = ".py" if lang == "python" else ".java"
                buggy_files.update(f for f in files if f.endswith(ext))

            results.append({
                "repo": f"{owner}/{name}",
                "language": lang,
                "stars": repo["stargazers_count"],
                "bug_fix_commits": len(bug_commits),
                "buggy_files": list(buggy_files),
            })
            time.sleep(1)

    output_path = os.path.join(output_dir, "mined_repos.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_buggy = sum(len(r["buggy_files"]) for r in results)
    logger.info("Mining complete. %d repos, %d buggy files. Saved to %s",
                len(results), total_buggy, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/mining_config.yaml")
    parser.add_argument("--output", default="../../data/raw")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mine(cfg, args.output)
