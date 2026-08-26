#!/usr/bin/env bash
# run_pipeline.sh — Full data pipeline: scrape → train → predict
# Usage: ./run_pipeline.sh [league] [season]
# Example: ./run_pipeline.sh premier_league 2024-2025

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python3"
[ -f "$PYTHON" ] || PYTHON="python3"

LEAGUE="${1:-premier_league}"
SEASON="${2:-2024-2025}"
CLI="$SCRIPT_DIR/bet_neural_cli.py"

echo ""
echo "🏆 Bet Neural v2 — Full Pipeline"
echo "  League: $LEAGUE"
echo "  Season: $SEASON"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Scraping data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$CLI" scrape --league "$LEAGUE" --season "$SEASON"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Training ML models..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$CLI" train --league "$LEAGUE" --season "$SEASON"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Gameweek predictions..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON "$CLI" gameweek --league "$LEAGUE"

echo ""
echo "✅ Pipeline complete!"
