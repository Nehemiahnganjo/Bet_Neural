"""
scraper.py — Bet Neural Data Scraper Engine v3
===============================================
Updated to use football-data.org API as primary source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR,
           RAW_DIR / "api", RAW_DIR / "understat",
           RAW_DIR / "odds", RAW_DIR / "fixtures"):
    _d.mkdir(parents=True, exist_ok=True)

LEAGUE_META: Dict[str, Dict] = {
    "premier_league": {"name": "Premier League", "country": "England", "fd_code": "PL", "understat_name": "EPL"},
    "la_liga": {"name": "La Liga", "country": "Spain", "fd_code": "PD", "understat_name": "La_liga"},
    "bundesliga": {"name": "Bundesliga", "country": "Germany", "fd_code": "BL1", "understat_name": "Bundesliga"},
    "serie_a": {"name": "Serie A", "country": "Italy", "fd_code": "SA", "understat_name": "Serie_A"},
    "ligue_1": {"name": "Ligue 1", "country": "France", "fd_code": "FL1", "understat_name": "Ligue_1"},
    "eredivisie": {"name": "Eredivisie", "country": "Netherlands", "fd_code": "DED"},
    "primeira_liga": {"name": "Primeira Liga", "country": "Portugal", "fd_code": "PPL"},
}

CURRENT_SEASON = "2024-2025"


class FootballDataClient:
    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", '')
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str, params: Optional[Dict] = None, cache_minutes: int = 60) -> Dict:
        cache_key = path.replace('/', '_') + (json.dumps(params, sort_keys=True) if params else "")
        cache_file = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None

        if cache_file and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < cache_minutes * 60:
                with open(cache_file) as fh:
                    return json.load(fh)

        time.sleep(0.1)
        headers = {'X-Auth-Token': self.api_key} if self.api_key else {}
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code} for {url}")

        data = response.json()
        if cache_file:
            with open(cache_file, 'w') as fh:
                json.dump(data, fh, indent=2)
        return data

    def fetch_matches(self, league: str, status: str = 'FINISHED',
                     days_back: int = 180, limit: int = 500) -> List[Dict]:
        code = LEAGUE_META.get(league, {}).get("fd_code")
        if not code:
            return []

        date_from = (datetime.now().date() - timedelta(days=days_back)).isoformat()
        date_to = datetime.now().date().isoformat()

        matches = []
        params = {'status': status, 'dateFrom': date_from, 'dateTo': date_to, 'limit': 100}

        while len(matches) < limit:
            data = self._get(f"competitions/{code}/matches", params=params)
            batch = data.get('matches', [])
            if not batch:
                break

            for m in batch:
                score = m.get('score', {})
                winner = score.get('winner')
                if winner not in ('HOME_TEAM', 'DRAW', 'AWAY_TEAM'):
                    continue
                full_time = score.get('fullTime', {})
                hg = full_time.get('home')
                ag = full_time.get('away')
                matches.append({
                    'home_team':  _norm(m['homeTeam']['name']),
                    'away_team':  _norm(m['awayTeam']['name']),
                    'result':     {'HOME_TEAM': 'H', 'DRAW': 'D', 'AWAY_TEAM': 'A'}[winner],
                    'home_goals': hg,       # canonical field for features.py
                    'away_goals': ag,
                    'home_score': hg,       # alias kept for display
                    'away_score': ag,
                    'date_utc':   m.get('utcDate', ''),
                    'league':     league,
                })

            if len(batch) < 100:
                break
            next_link = data.get('_links', {}).get('next', {}).get('href')
            if not next_link:
                break
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(next_link)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        return matches[:limit]

    def fetch_standings(self, league: str) -> List[Dict]:
        code = LEAGUE_META.get(league, {}).get("fd_code")
        if not code:
            return []
        data = self._get(f"competitions/{code}/standings")
        table = []
        for group in data.get('standings', []):
            if group.get('type') != 'TOTAL':
                continue
            for row in group.get('table', []):
                table.append({
                    'position': row['position'],
                    'team': _norm(row['team']['name']),
                    'played': row['playedGames'],
                    'won': row['won'],
                    'drawn': row['draw'],
                    'lost': row['lost'],
                    'points': row['points'],
                    'gd': row['goalDifference'],
                })
        return table


def _norm(name: str) -> str:
    return name.lower().strip().replace('.', '').replace("'", "")


def _calc_result(home, away):
    if home is None or away is None:
        return None
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


class BetNeuralScraper:
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        self.fd_client = FootballDataClient(api_key, cache_dir)

    def scrape_league(self, league: str, season: str = CURRENT_SEASON, include_odds: bool = True, fetch_understat: bool = True) -> Dict:
        logger.info(f"=== Scraping {league} ({season}) ===")
        result = {"league": league, "season": season, "scraped_at": datetime.now().isoformat()}

        try:
            matches = self.fd_client.fetch_matches(league, days_back=180, limit=300)
            result['matches'] = matches
            self._save_json(f"api/{league}_{season}_matches.json", matches)
            logger.info(f"API matches: {len(matches)}")
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
            result['matches'] = self._mock_matches(league)

        try:
            standings = self.fd_client.fetch_standings(league)
            result['standings'] = standings
            self._save_json(f"api/{league}_standings.json", standings)
            logger.info(f"Standings: {len(standings)} teams")
        except Exception as e:
            logger.warning(f"Standings fetch failed: {e}")

        # Fetch Understat xG data if available
        if fetch_understat:
            understat_data = self.fetch_understat_xg(league, season)
            if understat_data:
                result['understat_matches'] = understat_data
                self._save_json(f"understat/{league}_{season}_understat.json", understat_data)
                logger.info(f"Understat xG matches: {len(understat_data)}")

        if not result.get('matches') or len(result['matches']) < 10:
            result['matches'] = self._mock_matches(league)
            result['mock_data'] = True
            logger.warning("Using mock data fallback")

        result['summary'] = {'matches': len(result.get('matches', [])), 'standings': len(result.get('standings', [])), 'understat_matches': len(result.get('understat_matches', [])), 'mock_data': result.get('mock_data', False)}
        logger.info(f"Scrape complete: {result['summary']}")
        return result

    def fetch_understat_xg(self, league: str, season: str = CURRENT_SEASON) -> List[Dict]:
        """
        Fetch xG data from Understat.com via web scraping.
        Returns player and team xG statistics.
        """
        import cloudscraper
        from fake_useragent import UserAgent
        import re
        import json
        
        logger.info(f"Fetching Understat xG data for {league} {season}...")
        
        # Mapping from our league keys to Understat URL paths
        understat_leagues = {
            "premier_league": "EPL",
            "la_liga": "La_liga",
            "bundesliga": "Bundesliga",
            "serie_a": "Serie_A",
            "ligue_1": "Ligue_1",
            "eredivisie": "Eredivisie",
            "primeira_liga": "Primeira_Liga",
        }
        
        understat_name = understat_leagues.get(league)
        if not understat_name:
            logger.warning(f"No Understat mapping for {league}")
            return []
        
        try:
            # Create scraper with realistic headers
            scraper = cloudscraper.create_scraper()
            ua = UserAgent()
            
            # Try the new Understat structure
            url = f"https://understat.com/{understat_name}/{season}"
            headers = {
                'User-Agent': ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = scraper.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Understat returned status {response.status_code}")
                return []
            
            # Look for JavaScript data arrays
            # New format: matchesData array
            matches_match = re.search(r'var\s+matchesData\s*=\s*(\[{.*?}\])', response.text, re.DOTALL)
            
            if matches_match:
                data_str = matches_match.group(1)
                try:
                    # Parse the JSON-like structure
                    json_str = data_str.replace("'", '"').replace('undefined', 'null')
                    data = json.loads(json_str)
                    logger.info(f"Understat xG data: {len(data)} matches found")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to parse Understat data: {e}")
            else:
                logger.warning("Could not find matchesData in Understat response")
                
        except Exception as e:
            logger.warning(f"Understat fetch failed: {e}")
        
        return []

    def _mock_matches(self, league: str) -> List[Dict]:
        import random
        teams = ['Arsenal', 'Liverpool', 'Man City', 'Spurs', 'Aston Villa',
                 'Newcastle', 'Man Utd', 'Brighton', 'West Ham', 'Chelsea']
        matches = []
        now = datetime.now()
        for i in range(30):
            date = now - timedelta(days=i * 3)
            home, away = random.choice(teams), random.choice(teams)
            while away == home:
                away = random.choice(teams)
            home_score, away_score = random.randint(0, 4), random.randint(0, 3)
            result = 'H' if home_score > away_score else ('A' if away_score > home_score else 'D')
            matches.append({
                'date_utc': date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'home_team': home,
                'away_team': away,
                'home_goals': home_score,
                'away_goals': away_score,
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                # Mock data has no xG — leave as None rather than inventing values.
                'xg_home': None,
                'xg_away': None,
                'league': league,
            })
        return matches

    def _save_json(self, path: str, data: Any) -> None:
        filepath = RAW_DIR / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as fh:
            json.dump(data, fh, indent=2)


# ── Datahub Scraper (CSV from datahub.io) ─────────────────────────────────────

class DatahubScraper:
    """Scrape football data from datahub.io CSV files for historical seasons."""
    
    DATAHUB_URLS = {
        "premier_league": "https://datahub.io/football/english-premier-league/_r/-/season-",
        "la_liga": "https://datahub.io/football/spanish-la-liga/_r/-/season-",
        "bundesliga": "https://datahub.io/football/german-bundesliga/_r/-/season-",
        "serie_a": "https://datahub.io/football/italian-serie-a/_r/-/season-",
        "ligue_1": "https://datahub.io/football/french-ligue-1/_r/-/season-",
        "eredivisie": "https://datahub.io/football/dutch-eredivisie/_r/-/season-",
        "primeira_liga": "https://datahub.io/football/portuguese-primeira-liga/_r/-/season-",
    }
    
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_csv_url(self, league: str, season: str) -> Optional[str]:
        """Build datahub.io CSV URL from league and season."""
        if league not in self.DATAHUB_URLS:
            return None
        try:
            start_year = season.split("-")[0][-2:]
            end_year = season.split("-")[1][-2:]
            season_code = f"{start_year}{end_year}"
            return f"{self.DATAHUB_URLS[league]}{season_code}.csv"
        except Exception:
            return None
    
    def fetch_season(self, league: str, season: str = "2023-2024") -> List[Dict]:
        """Fetch a full season from datahub.io."""
        import csv
        import io
        import requests
        
        url = self._get_csv_url(league, season)
        if not url:
            logger.warning(f"No datahub URL for {league}")
            return []
        
        logger.info(f"Fetching {league} {season} from datahub.io...")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed: {response.status_code}")
                return []
            
            reader = csv.DictReader(io.StringIO(response.text))
            matches = []
            
            for row in reader:
                try:
                    home_score = int(row.get('FTHG', 0))
                    away_score = int(row.get('FTAG', 0))
                    result = row.get('FTR', 'D')

                    matches.append({
                        'date': row.get('Date', ''),
                        'home_team': _norm(row.get('HomeTeam', '')),
                        'away_team': _norm(row.get('AwayTeam', '')),
                        'home_goals': home_score,
                        'away_goals': away_score,
                        'home_score': home_score,
                        'away_score': away_score,
                        'result': result if result in ('H', 'D', 'A') else 'D',
                        # datahub.io CSVs do not carry xG data.
                        # Setting to None prevents the feature pipeline from
                        # treating random noise as a real signal.  When xg_home /
                        # xg_away are None, MatchHistory.team_form will fall back
                        # to using goals-scored as a proxy.
                        'xg_home': None,
                        'xg_away': None,
                        'league': league,
                    })
                except Exception:
                    continue
            
            logger.info(f"Fetched {len(matches)} matches from datahub.io")
            return matches
            
        except Exception as e:
            logger.warning(f"Datahub fetch failed: {e}")
            return []
    
    def fetch_multiple_seasons(self, league: str, seasons: List[str]) -> List[Dict]:
        """Fetch multiple seasons and combine."""
        all_matches = []
        for season in seasons:
            matches = self.fetch_season(league, season)
            all_matches.extend(matches)
        logger.info(f"Total: {len(all_matches)} matches from {len(seasons)} seasons")
        return all_matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Bet Neural Scraper v3")
    parser.add_argument("command", choices=["scrape", "odds", "standings", "status"])
    parser.add_argument("--league", default="premier_league")
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("⚠️  Set FOOTBALL_DATA_API_KEY environment variable for full API access")
        print("   Current key: " + ("SET" if api_key else "NOT SET"))

    scraper = BetNeuralScraper(api_key=api_key, cache_dir=str(CACHE_DIR))

    if args.command == "scrape":
        leagues = list(LEAGUE_META.keys()) if args.all else [args.league]
        for lg in leagues:
            result = scraper.scrape_league(lg, args.season)
            print(f"\n✅ {lg}: {result['summary']}")

    elif args.command == "standings":
        try:
            standings = scraper.fd_client.fetch_standings(args.league)
            print(f"\n🏆 {args.league} Standings")
            print("  Pos  Team                    P    W    D    L    GD  Pts")
            print("  " + "-" * 60)
            for s in standings:
                print(f"  {s['position']:>3}  {s['team']:<25}  {s['played']:>3}  {s['won']:>3}  {s['drawn']:>3}  {s['lost']:>3}  {s['gd']:>+3}  {s['points']:>3}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.command == "status":
        print("\n📂 Cached data:")
        for f in sorted(RAW_DIR.rglob("*.json")):
            size = f.stat().st_size // 1024
            age_h = (time.time() - f.stat().st_mtime) / 3600
            print(f"  {f.relative_to(BASE_DIR)} ({size}KB, {age_h:.1f}h old)")

