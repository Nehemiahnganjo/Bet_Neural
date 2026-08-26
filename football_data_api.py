"""
football_data_api.py
====================
Football-Data.org API client for Bet Neural.

Fetches live and historical match data and feeds results back into the
BetNeuralPredictor Elo update loop.

API docs: https://www.football-data.org/documentation/quickstart
Free tier (Tier 1) allows ~10 requests/minute, returns 90 days of history.

Usage
-----
Set your API key in the environment (recommended) or pass it directly:

    export FOOTBALL_DATA_API_KEY=your_key_here

Then call from Python:

    from football_data_api import FootballDataClient
    from bet_neural import BetNeuralPredictor

    client  = FootballDataClient()              # reads key from env
    pred    = BetNeuralPredictor()
    updated = client.update_elo_from_recent_results(pred, league='premier_league')
    print(f"Updated {updated} matches")

Or from the command line:

    python3 football_data_api.py fetch --league premier_league
    python3 football_data_api.py update --league premier_league
    python3 football_data_api.py fixtures --league premier_league
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests not installed. Run: pip3 install requests")


# ── League codes ────────────────────────────────────────────────────────────────

def _load_dotenv(path: str) -> None:
    """
    Minimal .env loader — no external dependency required.
    Sets environment variables from KEY=VALUE lines.
    Already-set variables are not overwritten (os.environ takes priority).
    """
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


# Common suffixes appended by football-data.org that are absent from our
# Elo ratings store.  Stripped in the order listed (longest first matters
# for e.g. "AFC Bournemouth" → "Bournemouth" via prefix removal below).
_TEAM_SUFFIXES = [
    ' FC', ' AFC', ' SC', ' CF', ' AC', ' BC', ' FK', ' SK',
    ' United', ' City',   # only when appended *as suffix with no further word
]

# Prefixes that are part of the official API name but absent from our store
_TEAM_PREFIXES = [
    'AFC ', 'FC ', 'SC ', 'BSC ',
]

# Manual overrides for cases where automatic stripping produces the wrong name.
# Maps API name → Elo store name.
_TEAM_NAME_OVERRIDES: Dict[str, str] = {
    'AFC Bournemouth':              'Bournemouth',
    'Brighton & Hove Albion FC':    'Brighton',
    'Brighton & Hove Albion':       'Brighton',
    'Coventry City FC':             'Coventry City',
    'Hull City AFC':                'Hull City',
    'Ipswich Town FC':              'Ipswich Town',
    'Leeds United FC':              'Leeds United',
    'Manchester City FC':           'Manchester City',
    'Manchester United FC':         'Manchester United',
    'Newcastle United FC':          'Newcastle',
    'Nottingham Forest FC':         'Nottingham Forest',
    'Sunderland AFC':               'Sunderland',
    'Tottenham Hotspur FC':         'Tottenham',
    'Aston Villa FC':               'Aston Villa',
    'Chelsea FC':                   'Chelsea',
    'Crystal Palace FC':            'Crystal Palace',
    'Everton FC':                   'Everton',
    'Fulham FC':                    'Fulham',
    'Liverpool FC':                 'Liverpool',
    'Arsenal FC':                   'Arsenal',
    'West Ham United FC':           'West Ham',
    'Wolverhampton Wanderers FC':   'Wolves',
    'Brentford FC':                 'Brentford',
    # La Liga
    'Club Atlético de Madrid':      'Atletico Madrid',
    'Athletic Club':                'Athletic Bilbao',
    'Real Betis Balompié':          'Betis',
    'Villarreal CF':                'Villarreal',
    'Real Sociedad de Fútbol':      'Real Sociedad',
    'Valencia CF':                  'Valencia',
    'Sevilla FC':                   'Sevilla',
    # Bundesliga
    'Borussia Dortmund':            'Borussia Dortmund',
    'FC Bayern München':            'Bayern Munich',
    'Bayer 04 Leverkusen':          'Bayer Leverkusen',
    'RasenBallsport Leipzig':       'RB Leipzig',
    'Eintracht Frankfurt':          'Eintracht Frankfurt',
    'VfL Wolfsburg':                'Wolfsburg',
    '1. FC Union Berlin':           'Union Berlin',
    'SC Freiburg':                  'Freiburg',
    'Borussia Mönchengladbach':     'Borussia Monchengladbach',
    'FC Augsburg':                  'Augsburg',
    # Serie A
    'FC Internazionale Milano':     'Inter Milan',
    'Juventus FC':                  'Juventus',
    'AC Milan':                     'AC Milan',
    'SSC Napoli':                   'Napoli',
    'AS Roma':                      'Roma',
    'SS Lazio':                     'Lazio',
    'Atalanta BC':                  'Atalanta',
    'ACF Fiorentina':               'Fiorentina',
    'Bologna FC 1909':              'Bologna',
    'Torino FC':                    'Torino',
    # Ligue 1
    'Paris Saint-Germain FC':       'PSG',
    'Olympique de Marseille':       'Marseille',
    'AS Monaco FC':                 'Monaco',
    'Olympique Lyonnais':           'Lyon',
    'LOSC Lille':                   'Lille',
    'OGC Nice':                     'Nice',
    'RC Lens':                      'Lens',
    'Stade Rennais FC 1901':        'Rennes',
    'RC Strasbourg Alsace':         'Strasbourg',
    'FC Nantes':                    'Nantes',
    # Eredivisie
    'AFC Ajax':                     'Ajax',
    'PSV Eindhoven':                'PSV',
    'Feyenoord Rotterdam':          'Feyenoord',
    'AZ Alkmaar':                   'AZ Alkmaar',
    'FC Twente':                    'Twente',
    'FC Utrecht':                   'Utrecht',
    'SBV Vitesse':                  'Vitesse',
    'SC Heerenveen':                'Heerenveen',
    'FC Groningen':                 'Groningen',
    'Sparta Rotterdam':             'Sparta Rotterdam',
    # Primeira Liga
    'Sport Lisboa e Benfica':       'Benfica',
    'FC Porto':                     'Porto',
    'Sporting CP':                  'Sporting CP',
    'SC Braga':                     'Braga',
    'Vitória SC':                   'Guimaraes',
    'CF Os Belenenses':             'Boavista',
}


def _normalise_team_name(name: str) -> str:
    """
    Convert a football-data.org team name to the name used in the Elo store.

    Resolution order:
      1. Exact override lookup
      2. Strip known prefixes / suffixes and retry override lookup
      3. Return the stripped version (fuzzy matcher in bet_neural.py handles
         anything still not found)
    """
    if name in _TEAM_NAME_OVERRIDES:
        return _TEAM_NAME_OVERRIDES[name]

    cleaned = name

    # Strip prefix
    for prefix in _TEAM_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    # Strip suffix
    for suffix in _TEAM_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    cleaned = cleaned.strip()

    # Retry override lookup after stripping
    if cleaned in _TEAM_NAME_OVERRIDES:
        return _TEAM_NAME_OVERRIDES[cleaned]

    return cleaned


LEAGUE_CODES: Dict[str, str] = {    'premier_league': 'PL',
    'la_liga':        'PD',
    'bundesliga':     'BL1',
    'serie_a':        'SA',
    'ligue_1':        'FL1',
    'eredivisie':     'DED',
    'primeira_liga':  'PPL',
}

# football-data.org result codes → Bet Neural result codes
_FD_RESULT_MAP = {'HOME_TEAM': 'H', 'DRAW': 'D', 'AWAY_TEAM': 'A'}


# ── Mock fixture generator ───────────────────────────────────────────────────

# Realistic team pools per league (kept in sync with bet_neural_lite.py)
_LEAGUE_TEAMS: Dict[str, List[str]] = {
    'premier_league': [
        'Arsenal', 'Manchester City', 'Liverpool', 'Chelsea',
        'Manchester United', 'Tottenham', 'Newcastle', 'Brighton',
        'Aston Villa', 'West Ham', 'Crystal Palace', 'Fulham',
    ],
    'la_liga': [
        'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
        'Villarreal', 'Real Sociedad', 'Athletic Bilbao', 'Valencia',
        'Betis', 'Osasuna',
    ],
    'bundesliga': [
        'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
        'Eintracht Frankfurt', 'Wolfsburg', 'Union Berlin', 'Freiburg',
        'Borussia Monchengladbach', 'Augsburg',
    ],
    'serie_a': [
        'Inter Milan', 'Juventus', 'AC Milan', 'Napoli', 'Roma',
        'Lazio', 'Atalanta', 'Fiorentina', 'Bologna', 'Torino',
    ],
    'ligue_1': [
        'PSG', 'Marseille', 'Monaco', 'Lyon', 'Lille',
        'Nice', 'Lens', 'Rennes', 'Strasbourg', 'Nantes',
    ],
    'eredivisie': [
        'Ajax', 'PSV', 'Feyenoord', 'AZ Alkmaar', 'Twente',
        'Utrecht', 'Vitesse', 'Heerenveen', 'Groningen', 'Sparta Rotterdam',
    ],
    'primeira_liga': [
        'Benfica', 'Porto', 'Sporting CP', 'Braga', 'Guimaraes',
        'Santa Clara', 'Famalicao', 'Gil Vicente', 'Boavista', 'Moreirense',
    ],
}

# Typical European kick-off slots (UTC hour)
_KICKOFF_SLOTS: Dict[str, List[int]] = {
    'premier_league': [12, 14, 15, 17],   # Sat/Sun UTC
    'la_liga':        [17, 19, 20, 21],
    'bundesliga':     [13, 15, 17, 19],
    'serie_a':        [17, 19, 20],
    'ligue_1':        [17, 19, 20],
    'eredivisie':     [14, 16, 18],
    'primeira_liga':  [18, 20, 22],
}


def generate_mock_fixtures(
    league: str = 'premier_league',
    days_ahead: int = 7,
    matches_per_day: int = 3,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate a realistic set of upcoming fixtures for *league* spread across
    the next *days_ahead* days.

    The same seed produces the same fixture list, so the board doesn't
    shuffle on every refresh.  The default seed is derived from the current
    ISO week number, so fixtures "rotate" once per week.
    """
    import random as _random

    teams = _LEAGUE_TEAMS.get(league, _LEAGUE_TEAMS['premier_league'])
    slots = _KICKOFF_SLOTS.get(league, [15, 17, 20])

    if seed is None:
        # Stable within the current week; changes each Monday
        seed = int(datetime.now().strftime('%Y%W')) + abs(hash(league)) % 1000

    rng = _random.Random(seed)

    # Shuffle teams into a randomised pairing order for this week
    shuffled = teams[:]
    rng.shuffle(shuffled)

    # Pair them up (round-robin style, wrap if odd number of teams)
    pairs: List[tuple] = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    # Add one more fixture if we have leftover pairs to fill the week
    if len(pairs) < days_ahead * matches_per_day:
        extra = teams[:]
        rng.shuffle(extra)
        for i in range(0, len(extra) - 1, 2):
            if len(pairs) >= days_ahead * matches_per_day:
                break
            h, a = extra[i], extra[i + 1]
            # Avoid re-using the same pair or same-team fixtures
            if h != a and (h, a) not in pairs and (a, h) not in pairs:
                pairs.append((h, a))

    now = datetime.now(tz=__import__('datetime').timezone.utc).replace(tzinfo=None)
    fixtures: List[Dict] = []

    for day_offset in range(days_ahead):
        day = now.date() + timedelta(days=day_offset)
        day_slots = rng.sample(slots, min(matches_per_day, len(slots)))
        day_slots.sort()

        for slot_hour in day_slots:
            if not pairs:
                break
            home, away = pairs.pop(0)
            kickoff = datetime(day.year, day.month, day.day, slot_hour, 0, 0)
            # Skip fixtures that are already in the past
            if kickoff < now:
                kickoff += timedelta(days=1)

            fixtures.append({
                'home_team': home,
                'away_team': away,
                'date_utc':  kickoff.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'league':    league,
                'mock':      True,   # flag so the GUI can show a ⚙ indicator
            })

        if not pairs:
            break

    fixtures.sort(key=lambda f: f['date_utc'])
    return fixtures


class FootballDataClient:
    """
    Thin client for the football-data.org v4 REST API.

    Parameters
    ----------
    api_key : str | None
        API key.  Falls back to the ``FOOTBALL_DATA_API_KEY`` environment
        variable.  If neither is set, requests will hit the unauthenticated
        endpoint (very limited).
    rate_limit_delay : float
        Seconds to wait between requests to respect the free-tier rate limit
        (≤ 10 req/min).  Default 6 s.
    cache_dir : str | None
        Optional directory to cache raw API responses as JSON files,
        reducing redundant network calls during development.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 6.0,
        cache_dir: Optional[str] = None,
    ) -> None:
        # Load .env from the project directory if it exists (before env lookup)
        _load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

        self.api_key = api_key or os.environ.get('FOOTBALL_DATA_API_KEY', '')
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir
        self._last_request_time: float = 0.0

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Low-level HTTP                                                      #
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a GET request, honouring the rate limit and optional cache.

        Raises RuntimeError on non-200 responses.
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not installed.")

        cache_key = path.replace('/', '_').strip('_')
        if params:
            cache_key += '_' + '_'.join(f"{k}{v}" for k, v in sorted(params.items()))
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json") if self.cache_dir else None

        if cache_file and os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 3600:          # cache valid for 1 hour
                with open(cache_file) as fh:
                    return json.load(fh)

        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        headers = {'X-Auth-Token': self.api_key} if self.api_key else {}
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        self._last_request_time = time.time()

        if response.status_code == 429:
            # Respect retry-after header if present
            retry_after = int(response.headers.get('X-RequestCounter-Reset', 60))
            print(f"⚠️  Rate limited. Waiting {retry_after}s …")
            time.sleep(retry_after)
            return self._get(path, params)

        if response.status_code != 200:
            raise RuntimeError(
                f"API error {response.status_code} for {url}: {response.text[:200]}"
            )

        data = response.json()

        if cache_file:
            with open(cache_file, 'w') as fh:
                json.dump(data, fh, indent=2)

        return data

    # ------------------------------------------------------------------ #
    #  Public API methods                                                  #
    # ------------------------------------------------------------------ #

    def fetch_finished_matches(
        self,
        league: str = 'premier_league',
        days_back: int = 180,
        max_matches: int = 500,
    ) -> List[Dict]:
        """
        Return a list of finished matches from the past *days_back* days.
        Fetches in batches to get up to *max_matches* results.
        """
        code = LEAGUE_CODES.get(league)
        if not code:
            raise ValueError(f"Unknown league: {league!r}. Choices: {list(LEAGUE_CODES)}")

        matches = []
        date_from = (date.today() - timedelta(days=days_back)).isoformat()
        date_to = date.today().isoformat()

        # Fetch in batches of 100 matches
        params = {
            'status': 'FINISHED',
            'dateFrom': date_from,
            'dateTo': date_to,
            'limit': 100
        }
        
        while len(matches) < max_matches:
            data = self._get(f"competitions/{code}/matches", params=params)
            batch = data.get('matches', [])
            
            if not batch:
                break
                
            for m in batch:
                score = m.get('score', {})
                winner = score.get('winner')
                if winner not in _FD_RESULT_MAP:
                    continue

                full_time = score.get('fullTime', {})
                hg = full_time.get('home')
                ag = full_time.get('away')
                matches.append({
                    'home_team':   _normalise_team_name(m['homeTeam']['name']),
                    'away_team':   _normalise_team_name(m['awayTeam']['name']),
                    'result':      _FD_RESULT_MAP[winner],
                    'home_goals':  hg,        # canonical field for features.py
                    'away_goals':  ag,
                    'home_score':  hg,        # alias kept for CLI display
                    'away_score':  ag,
                    'date_utc':    m.get('utcDate', ''),
                    'league':      league,
                })
            
            # Check if we need to paginate
            if len(batch) < 100:
                break
                
            # Get next page
            next_page = data.get('_links', {}).get('next', {}).get('href')
            if not next_page:
                break
                
            # Extract the next page params from URL
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(next_page)
            params = parse_qs(parsed.query)
            # Convert from list to single values
            params = {k: v[0] for k, v in params.items()}
        
        return matches[:max_matches]

    def fetch_upcoming_fixtures(
        self,
        league: str = 'premier_league',
        days_ahead: int = 7,
    ) -> List[Dict]:
        """
        Return scheduled (unplayed) fixtures for the next *days_ahead* days.

        Each item contains: home_team, away_team, date_utc, league

        Falls back to mock data automatically when no API key is set.
        """
        if not self.api_key:
            return generate_mock_fixtures(league, days_ahead)
        code = LEAGUE_CODES.get(league)
        if not code:
            raise ValueError(f"Unknown league: {league!r}.")

        date_from = date.today().isoformat()
        date_to   = (date.today() + timedelta(days=days_ahead)).isoformat()

        try:
            data = self._get(
                f"competitions/{code}/matches",
                params={'status': 'SCHEDULED', 'dateFrom': date_from, 'dateTo': date_to},
            )
        except RuntimeError:
            # API call failed (e.g. invalid key, network error) — use mock
            return generate_mock_fixtures(league, days_ahead)

        fixtures = []
        for m in data.get('matches', []):
            fixtures.append({
                'home_team': _normalise_team_name(m['homeTeam']['name']),
                'away_team': _normalise_team_name(m['awayTeam']['name']),
                'date_utc':  m.get('utcDate', ''),
                'league':    league,
            })

        # If API returned nothing meaningful, fall back to mock
        return fixtures if fixtures else generate_mock_fixtures(league, days_ahead)

    def fetch_standings(self, league: str = 'premier_league') -> List[Dict]:
        """
        Return the current league table as a list of team-standing dicts.

        Each item: position, team, played, won, drawn, lost, points, goal_diff
        """
        code = LEAGUE_CODES.get(league)
        if not code:
            raise ValueError(f"Unknown league: {league!r}.")

        data = self._get(f"competitions/{code}/standings")
        table_entries = []

        for standing_group in data.get('standings', []):
            if standing_group.get('type') != 'TOTAL':
                continue
            for row in standing_group.get('table', []):
                table_entries.append({
                    'position':  row['position'],
                    'team':      row['team']['name'],
                    'played':    row['playedGames'],
                    'won':       row['won'],
                    'drawn':     row['draw'],
                    'lost':      row['lost'],
                    'points':    row['points'],
                    'goal_diff': row['goalDifference'],
                })

        return table_entries

    # ------------------------------------------------------------------ #
    #  Elo update integration                                              #
    # ------------------------------------------------------------------ #

    def update_elo_from_recent_results(
        self,
        predictor,                          # BetNeuralPredictor instance
        league: str = 'premier_league',
        days_back: int = 30,
    ) -> int:
        """
        Fetch finished matches and update the predictor's Elo ratings.

        Returns the number of matches processed.

        Note: results are processed in chronological order so each match
        builds on correctly updated ratings.
        """
        print(f"🔄 Fetching {league} results (last {days_back} days) …")
        matches = self.fetch_finished_matches(league, days_back)

        if not matches:
            print("ℹ️  No finished matches found in the requested window.")
            return 0

        # Sort by date (oldest first)
        matches.sort(key=lambda m: m['date_utc'])

        for i, m in enumerate(matches, 1):
            old_home = predictor.get_team_elo(m['home_team'], league)
            old_away = predictor.get_team_elo(m['away_team'], league)

            new_home, new_away = predictor.update_after_match(
                m['home_team'], m['away_team'], m['result'], league, persist=False
            )

            print(
                f"  [{i:>3}] {m['date_utc'][:10]}  "
                f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}  "
                f"({m['result']})  |  "
                f"Elo: {old_home:.0f}→{new_home:.0f}  /  {old_away:.0f}→{new_away:.0f}"
            )

        # One save at the end rather than after every match
        predictor.save_ratings()
        print(f"✅ Processed {len(matches)} matches. Ratings saved.")
        return len(matches)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _print_usage() -> None:
    print("""
football_data_api.py  –  Bet Neural data pipeline
===================================================

USAGE:
  python3 football_data_api.py <command> [--league <league>] [--days <n>]

COMMANDS:
  fetch       Fetch & display recent finished results
  update      Fetch results AND update Elo ratings in elo_ratings.json
  fixtures    Show upcoming fixtures for the next 7 days
  standings   Show current league table

OPTIONS:
  --league    League key (default: premier_league)
              Choices: premier_league, la_liga, bundesliga,
                       serie_a, ligue_1, eredivisie, primeira_liga
  --days      Days back for fetch/update (default: 30)

ENVIRONMENT:
  FOOTBALL_DATA_API_KEY   Your football-data.org API token

EXAMPLE:
  FOOTBALL_DATA_API_KEY=abc123 python3 football_data_api.py update --league la_liga
""")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help', 'help'):
        _print_usage()
        return

    command = args[0]
    league  = 'premier_league'
    days    = 30

    i = 1
    while i < len(args):
        if args[i] == '--league' and i + 1 < len(args):
            league = args[i + 1].lower().replace(' ', '_')
            i += 2
        elif args[i] == '--days' and i + 1 < len(args):
            days = int(args[i + 1])
            i += 2
        else:
            i += 1

    client = FootballDataClient(
        cache_dir=os.path.join(os.path.dirname(__file__), '.api_cache')
    )

    if command == 'fetch':
        matches = client.fetch_finished_matches(league, days)
        print(f"\n📋 {league.replace('_', ' ').title()} — last {days} days ({len(matches)} matches)\n")
        for m in matches:
            print(
                f"  {m['date_utc'][:10]}  "
                f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}  "
                f"({m['result']})"
            )

    elif command == 'update':
        # Import here to avoid circular import when used as a library
        sys.path.insert(0, os.path.dirname(__file__))
        from bet_neural import BetNeuralPredictor
        predictor = BetNeuralPredictor()
        n = client.update_elo_from_recent_results(predictor, league, days)
        print(f"\n✅ Elo ratings updated from {n} matches.")

    elif command == 'fixtures':
        fixtures = client.fetch_upcoming_fixtures(league, days_ahead=days)
        print(f"\n📅 {league.replace('_', ' ').title()} — next {days} days ({len(fixtures)} fixtures)\n")
        for f in fixtures:
            # Format: "2026-08-25 15:00  Arsenal vs Chelsea [⚙]"
            # date_utc is "2026-08-25T15:00:00Z" for mock, "2026-08-25" for API
            date_utc = f['date_utc']
            if 'T' in date_utc:
                date_str = date_utc[:10]
                time_str = date_utc[11:16]
            else:
                date_str = date_utc[:10]
                time_str = "00:00"
            mock_flag = " ⚙" if f.get('mock') else ""
            print(f"  {date_str} {time_str}  {f['home_team']} vs {f['away_team']}{mock_flag}")

    elif command == 'standings':
        table = client.fetch_standings(league)
        print(f"\n🏆 {league.replace('_', ' ').title()} — Current Standings\n")
        print(f"  {'Pos':>3}  {'Team':<30} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GD':>4} {'Pts':>4}")
        print(f"  {'-'*60}")
        for row in table:
            print(
                f"  {row['position']:>3}  {row['team']:<30} "
                f"{row['played']:>3} {row['won']:>3} {row['drawn']:>3} {row['lost']:>3} "
                f"{row['goal_diff']:>+4} {row['points']:>4}"
            )

    else:
        print(f"❌ Unknown command: {command!r}")
        _print_usage()


if __name__ == '__main__':
    main()
