# 🏟️ Bet Neural — Real Match Data Collection Prompt

## OBJECTIVE
Collect authentic European football match data for **2024–2026** to populate the Bet Neural prediction system with real team histories, Elo ratings, and betting odds.

---

## DATA REQUIREMENTS

### Period
**January 1, 2024 → August 26, 2026** (inclusive)

### Leagues (Priority Order)
1. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Premier League** (England) — PL
2. 🇪🇸 **La Liga** (Spain) — PD  
3. 🇩🇪 **Bundesliga** (Germany) — BL1
4. 🇮🇹 **Serie A** (Italy) — SA
5. 🇫🇷 **Ligue 1** (France) — FL1
6. 🇳🇱 **Eredivisie** (Netherlands) — DED
7. 🇵🇹 **Primeira Liga** (Portugal) — PPL

---

## JSON FORMAT — Match Records

Each match record **MUST** follow this exact schema:

```json
{
  "id": "unique_identifier_string",
  "league": "premier_league|la_liga|bundesliga|serie_a|ligue_1|eredivisie|primeira_liga",
  "season": "2024-2025",
  "date_utc": "2024-08-17T15:00:00Z",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_goals": 2,
  "away_goals": 1,
  "home_score": 2,
  "away_score": 1,
  "result": "H|D|A",
  "xg_home": 2.15,
  "xg_away": 0.87,
  "shots_home": 12,
  "shots_away": 5,
  "shots_on_target_home": 5,
  "shots_on_target_away": 2,
  "passes_home": 487,
  "passes_away": 312,
  "possession_home": 62.5,
  "possession_away": 37.5,
  "fouls_home": 8,
  "fouls_away": 12,
  "yellow_cards_home": 2,
  "yellow_cards_away": 1,
  "red_cards_home": 0,
  "red_cards_away": 0,
  "corners_home": 6,
  "corners_away": 2,
  "venue": "Emirates Stadium",
  "attendance": 60432,
  "referee": "Andre Marriner"
}
```

### Field Explanations

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | string | ✅ | Unique ID (e.g., `fd_12345` from football-data.org) |
| `league` | string | ✅ | Normalized league code (see lookup below) |
| `season` | string | ✅ | Format: `YYYY-YYYY` (e.g., `2024-2025`) |
| `date_utc` | ISO 8601 | ✅ | UTC timestamp with Z suffix |
| `home_team` | string | ✅ | Team name (normalized: lowercase, no accents, no dots) |
| `away_team` | string | ✅ | Team name (normalized: same as home_team) |
| `home_goals` | int | ✅ | Canonical field — final score |
| `away_goals` | int | ✅ | Canonical field — final score |
| `home_score` | int | ✅ | Alias for display (same as home_goals) |
| `away_score` | int | ✅ | Alias for display (same as away_goals) |
| `result` | char | ✅ | `H` (home win), `D` (draw), `A` (away win) |
| `xg_home` | float | ⭐ | Expected goals home; if unavailable, use `None` |
| `xg_away` | float | ⭐ | Expected goals away; if unavailable, use `None` |
| `shots_home` | int | ❌ | Total shots (optional) |
| `shots_away` | int | ❌ | Total shots (optional) |
| `shots_on_target_home` | int | ❌ | Shots on target (optional) |
| `shots_on_target_away` | int | ❌ | Shots on target (optional) |
| `passes_home` | int | ❌ | Pass count (optional) |
| `passes_away` | int | ❌ | Pass count (optional) |
| `possession_home` | float | ❌ | Ball possession %; use decimal 0–100 (optional) |
| `possession_away` | float | ❌ | Ball possession %; use decimal 0–100 (optional) |
| `fouls_home` | int | ❌ | Fouls committed (optional) |
| `fouls_away` | int | ❌ | Fouls committed (optional) |
| `yellow_cards_home` | int | ❌ | Yellow cards (optional) |
| `yellow_cards_away` | int | ❌ | Yellow cards (optional) |
| `red_cards_home` | int | ❌ | Red cards (optional) |
| `red_cards_away` | int | ❌ | Red cards (optional) |
| `corners_home` | int | ❌ | Corners won (optional) |
| `corners_away` | int | ❌ | Corners won (optional) |
| `venue` | string | ❌ | Stadium name (optional) |
| `attendance` | int | ❌ | Match attendance (optional) |
| `referee` | string | ❌ | Match referee (optional) |

### Legend
- ✅ **Required**: Must be present in every record
- ⭐ **Strongly recommended**: Use if available; otherwise `None` or omit
- ❌ **Optional**: Include if available, no penalty if missing

---

## Team Name Normalization

Apply this transformation to all team names:
1. Convert to **lowercase**
2. Strip leading/trailing whitespace
3. Remove periods (`.`) → `manchester city fc` not `man. city f.c.`
4. Remove apostrophes (`'`) → `old trafford` not `o'reilly park`
5. Remove special characters (accents, tildes)

**Examples:**
- `"FC Barcelona"` → `"barcelona"`
- `"Paris Saint-Germain"` → `"paris saint-germain"`
- `"Real Madrid C.F."` → `"real madrid"`
- `"Manchester City"` → `"manchester city"`

---

## Data Sources (Recommended Priority)

### 1. **football-data.org** (Recommended)
- ✅ Covers all 7 leagues
- ✅ Reliable API with free tier
- ✅ Includes xG, possession, shots, fouls
- 📍 **API Key Required** (get free one at https://www.football-data.org)
- 📊 **Data**: 2024 onwards

**Usage:**
```bash
curl -H "X-Auth-Token: YOUR_API_KEY" \
  "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED&from=2024-01-01&to=2026-08-26"
```

### 2. **Understat** (xG specialist)
- ✅ Best xG data (shot-by-shot model)
- ⚠️ No free API; requires web scraping
- 📊 **Data**: 2024 onwards

### 3. **ESPN / Flashscore** (Last resort)
- ✅ Public data, easy to scrape
- ❌ Less structured; limited stats
- 📊 **Data**: Full history

---

## Output File Structure

Save collected data as **JSON Line** format (one match per line) in:
```
/home/void/Desktop/Bet_Neural/data/raw/api/matches_2024_2026.jsonl
```

**Example file:**
```jsonl
{"id": "fd_12345", "league": "premier_league", "season": "2024-2025", "date_utc": "2024-08-17T15:00:00Z", "home_team": "arsenal", "away_team": "chelsea", "home_goals": 2, "away_goals": 1, "result": "H", "xg_home": 2.15, "xg_away": 0.87}
{"id": "fd_12346", "league": "premier_league", "season": "2024-2025", "date_utc": "2024-08-18T15:00:00Z", "home_team": "manchester city", "away_team": "west ham united", "home_goals": 3, "away_goals": 1, "result": "H", "xg_home": 2.87, "xg_away": 0.92}
```

---

## Validation Checklist

Before submitting data, verify:

- [ ] **All required fields present** (id, league, season, date_utc, home_team, away_team, home_goals, away_goals, result)
- [ ] **No null values in required fields**
- [ ] **Date format**: ISO 8601 with `Z` suffix (e.g., `2024-08-17T15:00:00Z`)
- [ ] **Team names normalized**: lowercase, no dots/accents
- [ ] **League codes match**: `premier_league`, `la_liga`, `bundesliga`, `serie_a`, `ligue_1`, `eredivisie`, `primeira_liga`
- [ ] **Result field**: `H`, `D`, or `A` (matches outcome)
- [ ] **Goals match result**: If `home_goals > away_goals`, result must be `H`, etc.
- [ ] **Seasons covered**: Data from `2024-2025` and `2025-2026` (partial 2024-2026)
- [ ] **No duplicates**: Each match ID appears only once
- [ ] **xG values**: Either present (float > 0) or `None`/omitted, never negative

---

## Expected Coverage

**Target match count by league:**

| League | 2024-2025 | 2025-2026 (partial) | Total |
|--------|-----------|---------------------|-------|
| Premier League | 380 | 220 (est. through Aug 26) | ~600 |
| La Liga | 380 | 220 | ~600 |
| Bundesliga | 306 | 180 | ~486 |
| Serie A | 380 | 220 | ~600 |
| Ligue 1 | 380 | 220 | ~600 |
| Eredivisie | 306 | 180 | ~486 |
| Primeira Liga | 306 | 180 | ~486 |
| **TOTAL** | **2,438** | **1,420** | **~3,858** |

---

## Integration Steps (After Data Collection)

1. **Save to:** `/home/void/Desktop/Bet_Neural/data/raw/api/matches_2024_2026.jsonl`
2. **Run scraper:** 
   ```bash
   cd /home/void/Desktop/Bet_Neural
   python scraper.py load --source jsonl --path data/raw/api/matches_2024_2026.jsonl
   ```
3. **Verify import:**
   ```bash
   python -c "from scraper import FootballDataClient; c=FootballDataClient(); matches=c.fetch_matches('premier_league'); print(f'Loaded {len(matches)} matches')"
   ```
4. **Rebuild Elo ratings:**
   ```bash
   python bet_neural_cli.py train --all
   ```
5. **Test predictions:**
   ```bash
   python bet_neural_cli.py predict "Arsenal vs Chelsea" --league "Premier League"
   ```

---

## Notes

- **Privacy**: Do not collect or store personal player data beyond name/team affiliation
- **Rate limiting**: If using API, respect rate limits (usually 10 req/min free tier)
- **Caching**: The scraper automatically caches API responses for 60 minutes to avoid re-fetching
- **Incremental updates**: New data can be appended; the system auto-deduplicates by match ID
- **Timezone**: Always use **UTC** (`Z` suffix) for consistency

---

## Questions?

If data is missing or inconsistent:
- Check team name spelling (run through normalization)
- Verify league code matches list above
- Confirm date format (ISO 8601 with Z)
- Ensure no negative values in stats
- Check for duplicate match IDs

Good luck! 🎯
