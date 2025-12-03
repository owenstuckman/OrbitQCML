# src/data_extraction.py
"""
Extract PR data from GitHub repositories.
Handles API rate limiting, pagination, and data persistence.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from github import Github, RateLimitExceededException
from github.PullRequest import PullRequest
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    GITHUB_TOKEN, TARGET_REPOS, MIN_PR_LINES, 
    MAX_PRS_PER_REPO, LOOKBACK_DAYS, DATA_DIR
)


class GitHubExtractor:
    """Extract PR data from GitHub repositories."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or GITHUB_TOKEN
        if not self.token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN environment variable "
                "or pass token to constructor."
            )
        self.client = Github(self.token, per_page=100)
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _handle_rate_limit(self):
        """Wait for rate limit reset."""
        rate_limit = self.client.get_rate_limit()
        reset_time = rate_limit.core.reset
        # Make reset_time timezone-aware if it isn't
        if reset_time.tzinfo is None:
            reset_time = reset_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        sleep_time = (reset_time - now).total_seconds() + 10
        if sleep_time > 0:
            print(f"Rate limited. Sleeping for {sleep_time:.0f} seconds...")
            time.sleep(sleep_time)
    
    def _count_review_cycles(self, reviews) -> int:
        """Count number of review cycles (changes requested -> approved)."""
        cycles = 0
        pending_changes = False
        
        sorted_reviews = sorted(reviews, key=lambda r: r.submitted_at)
        
        for review in sorted_reviews:
            if review.state == "CHANGES_REQUESTED":
                pending_changes = True
            elif review.state == "APPROVED" and pending_changes:
                cycles += 1
                pending_changes = False
        
        return cycles
    
    def _extract_pr_record(self, pr: PullRequest, repo_name: str) -> Optional[dict]:
        """Extract data from a single PR."""
        try:
            # Skip if not merged or abandoned without review
            if pr.merged_at is None:
                return None
            
            # Get reviews
            reviews = list(pr.get_reviews())
            review_cycles = self._count_review_cycles(reviews)
            had_changes_requested = any(
                r.state == "CHANGES_REQUESTED" for r in reviews
            )
            
            # Get labels
            labels = [label.name for label in pr.labels]
            
            # Build record
            record = {
                "repo": repo_name,
                "pr_number": pr.number,
                "author": pr.user.login if pr.user else "unknown",
                "created_at": pr.created_at.isoformat(),
                "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
                "lines_added": pr.additions,
                "lines_deleted": pr.deletions,
                "files_changed": pr.changed_files,
                "commits": pr.commits,
                "review_comments": pr.review_comments,
                "comments": pr.comments,
                "review_cycles": review_cycles,
                "had_changes_requested": had_changes_requested,
                "labels": labels,
                "title": pr.title,
                "title_length": len(pr.title),
                "body_length": len(pr.body or ""),
                "is_draft": pr.draft if hasattr(pr, "draft") else False,
                # Target variable
                "passed_first_review": not had_changes_requested and review_cycles <= 1,
            }
            
            return record
            
        except Exception as e:
            print(f"Error extracting PR #{pr.number}: {e}")
            return None
    
    def extract_repo(
        self, 
        repo_name: str, 
        since_date: Optional[datetime] = None,
        max_prs: Optional[int] = None,
        checkpoint_every: int = 100,
    ) -> list[dict]:
        """
        Extract all PRs from a repository.
        
        Args:
            repo_name: GitHub repo in "owner/repo" format
            since_date: Only extract PRs updated after this date
            max_prs: Maximum PRs to extract
            checkpoint_every: Save checkpoint every N PRs (for crash recovery)
        """
        
        if since_date is None:
            since_date = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        
        max_prs = max_prs or MAX_PRS_PER_REPO
        
        # Check for partial checkpoint
        safe_name = repo_name.replace("/", "_")
        checkpoint_path = self.data_dir / f"checkpoint_{safe_name}.json"
        
        records = []
        last_pr_number = None
        
        if checkpoint_path.exists():
            print(f"\nFound checkpoint for {repo_name}, resuming...")
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            records = checkpoint.get("records", [])
            last_pr_number = checkpoint.get("last_pr_number")
            print(f"  Resuming from PR #{last_pr_number} ({len(records)} PRs already collected)")
        
        print(f"\nExtracting PRs from {repo_name}...")
        print(f"  Since: {since_date.date()}")
        print(f"  Max PRs: {max_prs or 'unlimited'}")
        
        try:
            repo = self.client.get_repo(repo_name)
        except Exception as e:
            print(f"Error accessing repo {repo_name}: {e}")
            return records  # Return any checkpointed records
        
        # Get closed PRs (includes merged)
        prs = repo.get_pulls(state="closed", sort="updated", direction="desc")
        
        pr_count = len(records)
        skipping = last_pr_number is not None
        
        with tqdm(desc=f"  Processing", unit=" PRs", initial=pr_count) as pbar:
            for pr in prs:
                # Check rate limit
                try:
                    # Skip until we reach where we left off
                    if skipping:
                        if pr.number == last_pr_number:
                            skipping = False
                        continue
                    
                    # Handle timezone-aware comparison
                    pr_updated = pr.updated_at
                    if pr_updated.tzinfo is None:
                        pr_updated = pr_updated.replace(tzinfo=timezone.utc)
                    
                    if pr_updated < since_date:
                        break
                    
                    # Skip small PRs
                    if pr.additions + pr.deletions < MIN_PR_LINES:
                        continue
                    
                    record = self._extract_pr_record(pr, repo_name)
                    if record:
                        records.append(record)
                        pr_count += 1
                        pbar.update(1)
                        
                        # Save checkpoint periodically
                        if pr_count % checkpoint_every == 0:
                            self._save_checkpoint(
                                checkpoint_path, records, pr.number
                            )
                    
                    if max_prs and pr_count >= max_prs:
                        break
                        
                except RateLimitExceededException:
                    # Save checkpoint before waiting
                    self._save_checkpoint(checkpoint_path, records, pr.number)
                    self._handle_rate_limit()
                    continue
                except KeyboardInterrupt:
                    # Save checkpoint on interrupt
                    print("\n\nInterrupted! Saving checkpoint...")
                    self._save_checkpoint(checkpoint_path, records, pr.number)
                    raise
                except Exception as e:
                    print(f"\nError on PR #{pr.number}: {e}")
                    continue
        
        # Clean up checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  Removed checkpoint file")
        
        print(f"  Extracted {len(records)} PRs")
        return records
    
    def _save_checkpoint(self, path: Path, records: list, last_pr: int):
        """Save extraction checkpoint for crash recovery."""
        checkpoint = {
            "records": records,
            "last_pr_number": last_pr,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f)
        # Don't print every time, too noisy
    
    def extract_all_repos(
        self, 
        repos: Optional[list[str]] = None,
        save_intermediate: bool = True,
        resume: bool = True,
    ) -> list[dict]:
        """
        Extract PRs from all target repositories.
        
        Args:
            repos: List of repos to extract from
            save_intermediate: Save each repo's data separately
            resume: Skip repos that already have cached data
        """
        repos = repos or TARGET_REPOS
        all_records = []
        
        for repo_name in repos:
            safe_name = repo_name.replace("/", "_")
            cache_path = self.data_dir / f"raw_{safe_name}.json"
            
            # Check for existing data if resume is enabled
            if resume and cache_path.exists():
                print(f"\nFound cached data for {repo_name}, loading...")
                with open(cache_path) as f:
                    records = json.load(f)
                print(f"  Loaded {len(records)} PRs from cache")
                all_records.extend(records)
                continue
            
            records = self.extract_repo(repo_name)
            all_records.extend(records)
            
            if save_intermediate:
                # Save per-repo data
                with open(cache_path, "w") as f:
                    json.dump(records, f, indent=2)
                print(f"  Saved to {cache_path}")
        
        # Save combined data
        output_path = self.data_dir / "raw_all_prs.json"
        with open(output_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nSaved {len(all_records)} total PRs to {output_path}")
        
        return all_records
    
    def load_cached_data(self) -> Optional[list[dict]]:
        """Load previously extracted data."""
        cache_path = self.data_dir / "raw_all_prs.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None


def should_include_pr(record: dict) -> bool:
    """Filter criteria for training data quality."""
    author = record.get("author", "").lower()
    
    # Exclude bots
    if "bot" in author:
        return False
    if author in ["dependabot", "renovate", "github-actions", "codecov"]:
        return False
    
    # Exclude trivial changes
    total_lines = record.get("lines_added", 0) + record.get("lines_deleted", 0)
    if total_lines < MIN_PR_LINES:
        return False
    
    # Exclude self-merges without review
    if record.get("review_comments", 0) == 0 and record.get("review_cycles", 0) == 0:
        # Check if there were any reviews at all
        if not record.get("had_changes_requested", False):
            # Might be auto-merged or self-merged
            pass  # Keep for now, model can learn from this
    
    return True


if __name__ == "__main__":
    # Run extraction
    extractor = GitHubExtractor()
    
    # Check for cached data
    cached = extractor.load_cached_data()
    if cached:
        print(f"Found cached data with {len(cached)} PRs")
        response = input("Re-extract? (y/N): ")
        if response.lower() != "y":
            print("Using cached data.")
            exit(0)
    
    # Extract fresh data
    records = extractor.extract_all_repos()
    
    # Apply filters
    filtered = [r for r in records if should_include_pr(r)]
    print(f"\nAfter filtering: {len(filtered)} PRs")
    
    # Save filtered data
    output_path = Path(DATA_DIR) / "filtered_prs.json"
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Saved filtered data to {output_path}")
