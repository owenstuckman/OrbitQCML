#!/usr/bin/env python3
# scripts/generate_demo_data.py
"""
Generate synthetic demo data for testing the pipeline without GitHub API access.
Useful for development and demonstration purposes.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR


def generate_author_pool(n_authors: int = 50) -> list[dict]:
    """Generate a pool of authors with varying skill levels."""
    authors = []
    
    skill_levels = ["novice", "junior", "regular", "senior", "expert"]
    skill_probs = [0.6, 0.7, 0.8, 0.9, 0.95]  # Base pass rates
    
    for i in range(n_authors):
        skill = random.choice(range(len(skill_levels)))
        authors.append({
            "name": f"author_{i:03d}",
            "skill_level": skill_levels[skill],
            "base_pass_rate": skill_probs[skill] + random.uniform(-0.1, 0.1),
            "avg_pr_size": random.randint(50, 500),
        })
    
    return authors


def generate_pr_record(
    author: dict,
    repo: str,
    pr_number: int,
    created_at: datetime,
) -> dict:
    """Generate a single synthetic PR record."""
    
    # Size based on author tendency + noise
    base_lines = author["avg_pr_size"]
    lines_added = max(10, int(np.random.lognormal(np.log(base_lines), 0.5)))
    lines_deleted = max(0, int(lines_added * random.uniform(0.1, 0.5)))
    files_changed = max(1, int(lines_added / random.randint(20, 100)))
    
    # Complexity affects pass rate
    complexity_factor = min(1.0, 100 / (lines_added + 1))
    
    # Documentation quality affects pass rate
    body_length = random.randint(0, 500) if random.random() > 0.3 else 0
    doc_factor = 0.1 if body_length > 100 else 0
    
    # Calculate pass probability
    pass_prob = author["base_pass_rate"] * complexity_factor + doc_factor
    pass_prob = max(0.1, min(0.99, pass_prob))
    
    # Determine if passed first review
    passed_first = random.random() < pass_prob
    
    # Generate review data based on outcome
    if passed_first:
        had_changes_requested = False
        review_cycles = 0
    else:
        had_changes_requested = True
        review_cycles = random.randint(1, 3)
    
    # Generate labels
    labels = []
    if random.random() < 0.3:
        labels.append("bug")
    if random.random() < 0.2:
        labels.append("feature")
    if random.random() < 0.1:
        labels.append("documentation")
    if random.random() < 0.15:
        labels.append("refactor")
    
    # Time to merge (if passed)
    merge_delay = timedelta(hours=random.randint(1, 72))
    merged_at = created_at + merge_delay
    
    return {
        "repo": repo,
        "pr_number": pr_number,
        "author": author["name"],
        "created_at": created_at.isoformat(),
        "merged_at": merged_at.isoformat(),
        "closed_at": merged_at.isoformat(),
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "files_changed": files_changed,
        "commits": random.randint(1, max(2, files_changed)),
        "review_comments": random.randint(0, 10) if had_changes_requested else random.randint(0, 3),
        "comments": random.randint(0, 5),
        "review_cycles": review_cycles,
        "had_changes_requested": had_changes_requested,
        "labels": labels,
        "title": f"PR {pr_number}: {'Fix' if 'bug' in labels else 'Add'} something",
        "title_length": random.randint(20, 80),
        "body_length": body_length,
        "is_draft": False,
        "passed_first_review": passed_first,
    }


def generate_demo_dataset(
    n_prs: int = 2000,
    n_authors: int = 50,
    n_repos: int = 3,
    start_date: datetime = None,
) -> list[dict]:
    """Generate a complete demo dataset."""
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    repos = [f"demo/repo-{i}" for i in range(n_repos)]
    authors = generate_author_pool(n_authors)
    
    records = []
    current_date = start_date
    
    for pr_num in range(n_prs):
        # Advance time
        current_date += timedelta(hours=random.randint(1, 12))
        
        # Select random author and repo
        author = random.choice(authors)
        repo = random.choice(repos)
        
        record = generate_pr_record(
            author=author,
            repo=repo,
            pr_number=pr_num + 1,
            created_at=current_date,
        )
        records.append(record)
    
    return records


def main():
    """Generate and save demo data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate demo data")
    parser.add_argument("--n-prs", type=int, default=2000, help="Number of PRs")
    parser.add_argument("--n-authors", type=int, default=50, help="Number of authors")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_prs} demo PRs from {args.n_authors} authors...")
    
    records = generate_demo_dataset(
        n_prs=args.n_prs,
        n_authors=args.n_authors,
    )
    
    # Calculate statistics
    pass_rate = sum(1 for r in records if r["passed_first_review"]) / len(records)
    avg_lines = sum(r["lines_added"] + r["lines_deleted"] for r in records) / len(records)
    
    print(f"\nGenerated dataset statistics:")
    print(f"  Total PRs: {len(records)}")
    print(f"  Pass rate: {pass_rate:.1%}")
    print(f"  Avg lines changed: {avg_lines:.0f}")
    print(f"  Unique authors: {len(set(r['author'] for r in records))}")
    
    # Save to file
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output or (data_dir / "filtered_prs.json")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print("\nYou can now run:")
    print("  python scripts/run_pipeline.py features")
    print("  python scripts/run_pipeline.py train --quick")


if __name__ == "__main__":
    main()
