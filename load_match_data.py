#!/usr/bin/env python3
"""
load_match_data.py — Ingest and validate real match data from JSONL
====================================================================

Usage:
    python load_match_data.py --input data/raw/api/matches_2024_2026.jsonl --output data/processed/matches.pkl
    python load_match_data.py --validate-only data/raw/api/matches_2024_2026.jsonl

Validates:
- All required fields present
- Team names normalized
- Date format (ISO 8601)
- No duplicates
- Result consistency (home_goals vs result field)
- Numeric bounds (goals, xG, possession)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pickle


VALID_LEAGUES = {
    "premier_league", "la_liga", "bundesliga", "serie_a",
    "ligue_1", "eredivisie", "primeira_liga"
}

VALID_SEASONS = {"2023-2024", "2024-2025", "2025-2026"}  # Accept 2023-2024 as historical context
REQUIRED_FIELDS = {
    "id", "league", "season", "date_utc", "home_team", "away_team",
    "home_goals", "away_goals", "result"
}

RESULT_MAP = {"H": (True, False), "D": (False, False), "A": (False, True)}


def normalize_team_name(name: str) -> str:
    """Normalize team name: lowercase, strip, remove dots/apostrophes."""
    return name.lower().strip().replace(".", "").replace("'", "")


def validate_match(record: Dict, row_num: int) -> Tuple[bool, List[str]]:
    """Validate a single match record. Returns (is_valid, errors)."""
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Row {row_num}: missing required field '{field}'")
        elif record[field] is None:
            errors.append(f"Row {row_num}: '{field}' is None")

    if errors:
        return False, errors

    # Validate league
    if record["league"] not in VALID_LEAGUES:
        errors.append(f"Row {row_num}: invalid league '{record['league']}'")

    # Validate season
    if record.get("season") not in VALID_SEASONS:
        errors.append(f"Row {row_num}: season '{record.get('season')}' not in {VALID_SEASONS}")

    # Validate date format
    try:
        dt = datetime.fromisoformat(record["date_utc"].replace("Z", "+00:00"))
        if dt.year < 2024 or dt.year > 2026:
            errors.append(f"Row {row_num}: date year {dt.year} outside 2024–2026 range")
    except (ValueError, AttributeError):
        errors.append(f"Row {row_num}: invalid date_utc format '{record.get('date_utc')}'")

    # Validate team names
    home_team = record.get("home_team", "")
    away_team = record.get("away_team", "")
    if not home_team or not away_team:
        errors.append(f"Row {row_num}: empty team name")
    elif home_team == away_team:
        errors.append(f"Row {row_num}: home_team == away_team '{home_team}'")
    elif not home_team.islower() or not away_team.islower():
        errors.append(f"Row {row_num}: team names must be lowercase")
    elif "." in home_team or "." in away_team:
        errors.append(f"Row {row_num}: team names contain dots (not normalized)")

    # Validate goals
    hg = record.get("home_goals")
    ag = record.get("away_goals")
    if not isinstance(hg, int) or not isinstance(ag, int):
        errors.append(f"Row {row_num}: goals must be integers, got {type(hg).__name__}, {type(ag).__name__}")
    elif hg < 0 or ag < 0:
        errors.append(f"Row {row_num}: negative goals {hg}-{ag}")

    # Validate result consistency
    result = record.get("result", "")
    if result not in ("H", "D", "A"):
        errors.append(f"Row {row_num}: invalid result '{result}' (must be H, D, or A)")
    else:
        home_wins, away_wins = RESULT_MAP[result]
        if home_wins and hg <= ag:
            errors.append(f"Row {row_num}: result is H but score is {hg}-{ag}")
        elif away_wins and ag <= hg:
            errors.append(f"Row {row_num}: result is A but score is {hg}-{ag}")
        elif result == "D" and hg != ag:
            errors.append(f"Row {row_num}: result is D but score is {hg}-{ag}")

    # Validate optional xG (if present, must be ≥ 0)
    for key in ("xg_home", "xg_away"):
        val = record.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            errors.append(f"Row {row_num}: {key} must be ≥ 0, got {val}")

    # Validate optional possession (if present, 0–100)
    for key in ("possession_home", "possession_away"):
        val = record.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < 0 or val > 100):
            errors.append(f"Row {row_num}: {key} must be 0–100, got {val}")

    return len(errors) == 0, errors


def load_and_validate(input_path: str, validate_only: bool = False) -> Tuple[List[Dict], int]:
    """
    Load JSONL, validate each record, and optionally save to pickle.
    Returns (records, error_count).
    """
    path = Path(input_path)
    if not path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    records = []
    errors_all = []
    seen_ids: Set[str] = set()
    league_counts = defaultdict(int)
    season_counts = defaultdict(int)

    print(f"\n📂 Loading from: {input_path}")
    print("=" * 70)

    with open(path) as fh:
        for row_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors_all.append(f"Row {row_num}: JSON parse error: {e}")
                continue

            # Validate
            is_valid, errs = validate_match(record, row_num)
            if not is_valid:
                errors_all.extend(errs)
                continue

            # Check for duplicates
            rid = record.get("id")
            if rid in seen_ids:
                errors_all.append(f"Row {row_num}: duplicate match ID '{rid}'")
                continue
            seen_ids.add(rid)

            # Normalize team names
            record["home_team"] = normalize_team_name(record["home_team"])
            record["away_team"] = normalize_team_name(record["away_team"])

            records.append(record)
            league_counts[record["league"]] += 1
            season_counts[record["season"]] += 1

    # Summary
    print(f"\n✅ Valid records: {len(records)}")
    print(f"❌ Errors found: {len(errors_all)}")
    print(f"🔑 Unique match IDs: {len(seen_ids)}")

    if records:
        print(f"\n📊 Coverage by league:")
        for league in sorted(league_counts.keys()):
            print(f"   {league:20s}: {league_counts[league]:4d} matches")

        print(f"\n📅 Coverage by season:")
        for season in sorted(season_counts.keys()):
            print(f"   {season}: {season_counts[season]:4d} matches")

    if errors_all:
        print(f"\n⚠️  First 20 errors:")
        for err in errors_all[:20]:
            print(f"   {err}")
        if len(errors_all) > 20:
            print(f"   ... and {len(errors_all) - 20} more")

    print("=" * 70)

    if not validate_only and records:
        print(f"\n💾 Saving to pickle...")
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "matches.pkl"
        with open(output_path, "wb") as fh:
            pickle.dump(records, fh)
        print(f"✅ Saved {len(records)} records to {output_path}")

    return records, len(errors_all)


def main():
    parser = argparse.ArgumentParser(
        description="Load and validate match data from JSONL"
    )
    parser.add_argument(
        "input",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate without saving to pickle"
    )
    parser.add_argument(
        "--output",
        help="Output pickle path (ignored if --validate-only)"
    )

    args = parser.parse_args()

    records, error_count = load_and_validate(args.input, validate_only=args.validate_only)

    if error_count > 0:
        print(f"\n⚠️  {error_count} validation errors found")
        sys.exit(1 if error_count > 10 else 0)
    else:
        print(f"\n✨ All {len(records)} records valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
