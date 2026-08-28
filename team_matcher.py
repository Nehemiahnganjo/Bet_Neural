"""
team_matcher.py — Enhanced Team Name Matching System
====================================================
Advanced fuzzy matching with multiple algorithms, team aliases,
and intelligent resolution for football team names.
"""

import difflib
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeamMatch:
    """Result of team matching operation."""
    matched_name: str
    similarity: float
    confidence: str  # 'exact', 'high', 'medium', 'low'
    method: str      # matching method used
    aliases_used: List[str]


class EnhancedTeamMatcher:
    """
    Multi-algorithm team name matching system with:
    - Common aliases and abbreviations
    - Multiple similarity algorithms  
    - League-specific team knowledge
    - Intelligent preprocessing
    """
    
    def __init__(self):
        self.team_aliases = self._build_team_aliases()
        self.common_words = {'fc', 'cf', 'united', 'city', 'real', 'athletic', 'club', 
                           'de', 'cf', 'sc', 'ac', 'inter', 'borussia', 'bayern'}
        
    def _build_team_aliases(self) -> Dict[str, Set[str]]:
        """Build comprehensive team alias database."""
        return {
            # Premier League
            'arsenal fc': {'arsenal', 'gunners', 'afc'},
            'chelsea fc': {'chelsea', 'blues', 'cfc'},
            'liverpool fc': {'liverpool', 'reds', 'lfc'},
            'manchester city fc': {'manchester city', 'man city', 'city', 'mcfc', 'citizens'},
            'manchester united fc': {'manchester united', 'man utd', 'man u', 'mufc', 'united'},
            'tottenham hotspur fc': {'tottenham', 'spurs', 'thfc'},
            'west ham united fc': {'west ham', 'hammers', 'whufc'},
            'newcastle united fc': {'newcastle', 'magpies', 'nufc'},
            'brighton & hove albion fc': {'brighton', 'seagulls', 'bhafc'},
            'aston villa fc': {'aston villa', 'villa', 'avfc'},
            'everton fc': {'everton', 'toffees', 'efc'},
            'crystal palace fc': {'crystal palace', 'palace', 'eagles', 'cpfc'},
            'brentford fc': {'brentford', 'bees', 'bfc'},
            'fulham fc': {'fulham', 'cottagers', 'ffc'},
            'wolverhampton wanderers fc': {'wolves', 'wolverhampton', 'wwfc'},
            'nottingham forest fc': {'nottingham forest', 'forest', 'nffc'},
            'afc bournemouth': {'bournemouth', 'cherries', 'afcb'},
            'sheffield united fc': {'sheffield united', 'blades', 'sufc'},
            'burnley fc': {'burnley', 'clarets', 'bfc'},
            'luton town fc': {'luton', 'hatters', 'ltfc'},
            
            # La Liga
            'real madrid cf': {'real madrid', 'madrid', 'real', 'rmcf', 'los blancos'},
            'fc barcelona': {'barcelona', 'barca', 'fcb', 'blaugrana'},
            'club atletico de madrid': {'atletico madrid', 'atletico', 'atleti', 'cad'},
            'athletic club': {'athletic bilbao', 'bilbao', 'athletic'},
            'real sociedad de futbol': {'real sociedad', 'sociedad', 'txuri-urdin'},
            'villarreal cf': {'villarreal', 'yellow submarine', 'vcf'},
            'real betis balompie': {'real betis', 'betis', 'beticos'},
            'sevilla fc': {'sevilla', 'sevillistas', 'sfc'},
            'valencia cf': {'valencia', 'che', 'vcf'},
            'deportivo alaves': {'alaves', 'babazorros'},
            'getafe cf': {'getafe', 'azulones', 'gcf'},
            'rayo vallecano de madrid': {'rayo vallecano', 'rayo', 'vallecano'},
            'rc celta de vigo': {'celta vigo', 'celta', 'celticos'},
            'ca osasuna': {'osasuna', 'rojillos'},
            'rcd mallorca': {'mallorca', 'bermellones'},
            'girona fc': {'girona', 'gfc'},
            'cadiz cf': {'cadiz', 'submarino amarillo'},
            'ud las palmas': {'las palmas', 'amarillos'},
            
            # Bundesliga  
            'fc bayern munchen': {'bayern munich', 'bayern', 'fcb', 'die roten'},
            'borussia dortmund': {'dortmund', 'bvb', 'die schwarzgelben'},
            'rb leipzig': {'leipzig', 'rbl', 'die roten bullen'},
            'bayer 04 leverkusen': {'leverkusen', 'bayer', 'die werkself'},
            'eintracht frankfurt': {'frankfurt', 'eintracht', 'sge'},
            'tsg 1899 hoffenheim': {'hoffenheim', 'tsg'},
            'sc freiburg': {'freiburg', 'scf'},
            'vfl wolfsburg': {'wolfsburg', 'die wolfe'},
            '1 fsv mainz 05': {'mainz', 'mainz 05', 'die nullfunfer'},
            'fc augsburg': {'augsburg', 'fca'},
            '1 fc union berlin': {'union berlin', 'union', 'eisern'},
            '1 fc koln': {'koln', 'cologne', 'fc koln', 'effzeh'},
            '1 fc heidenheim 1846': {'heidenheim', 'fch'},
            'sv darmstadt 98': {'darmstadt', 'sv98'},
            
            # Serie A
            'juventus fc': {'juventus', 'juve', 'bianconeri', 'jfc'},
            'ac milan': {'milan', 'rossoneri', 'acm'},
            'fc internazionale milano': {'inter milan', 'inter', 'nerazzurri', 'fcim'},
            'ssc napoli': {'napoli', 'partenopei', 'azzurri'},
            'as roma': {'roma', 'giallorossi', 'asr'},
            'ss lazio': {'lazio', 'biancocelesti', 'ssl'},
            'atalanta bc': {'atalanta', 'nerazzurri bergamaschi', 'abc'},
            'acf fiorentina': {'fiorentina', 'viola', 'acf'},
            'torino fc': {'torino', 'granata', 'tfc'},
            'bologna fc 1909': {'bologna', 'rossoblù', 'bfc'},
            'udinese calcio': {'udinese', 'bianconeri friulani', 'uc'},
            'us sassuolo calcio': {'sassuolo', 'neroverdi', 'usc'},
            'hellas verona fc': {'verona', 'gialloblu', 'hvfc'},
            'genoa cfc': {'genoa', 'rossoblù', 'gcfc'},
            'us salernitana 1919': {'salernitana', 'granata', 'uss'},
            'spezia calcio': {'spezia', 'aquilotti', 'sc'},
            'empoli fc': {'empoli', 'azzurri', 'efc'},
            'us lecce': {'lecce', 'giallorossi', 'usl'},
            'uc sampdoria': {'sampdoria', 'blucerchiati', 'ucs'},
            'cagliari calcio': {'cagliari', 'rossoblù', 'cc'},
            
            # Ligue 1
            'paris saint-germain fc': {'psg', 'paris sg', 'paris saint-germain'},
            'olympique de marseille': {'marseille', 'om', 'phocéens'},
            'olympique lyonnais': {'lyon', 'ol', 'gones'},
            'as monaco fc': {'monaco', 'asm', 'monégasques'},
            'stade rennais fc': {'rennes', 'srfc', 'rouge et noir'},
            'lille osc': {'lille', 'losc', 'dogues'},
            'rc lens': {'lens', 'rcl', 'sang et or'},
            'ogc nice': {'nice', 'ogcn', 'aiglons'},
            'stade de reims': {'reims', 'sdr'},
            'fc nantes': {'nantes', 'fcn', 'canaris'},
            'montpellier hsc': {'montpellier', 'mhsc'},
            'rc strasbourg alsace': {'strasbourg', 'rcsa'},
            'fc metz': {'metz', 'fcm'},
            'angers sco': {'angers', 'asco'},
            'stade brestois 29': {'brest', 'sb29'},
            'fc lorient': {'lorient', 'fcl'},
            
            # Eredivisie
            'afc ajax': {'ajax', 'ajax amsterdam', 'godenzonen'},
            'psv eindhoven': {'psv', 'boeren'},
            'feyenoord rotterdam': {'feyenoord', 'de club'},
            'fc twente': {'twente', 'tukkers'},
            'az alkmaar': {'az', 'kaaskoppen'},
            'fc utrecht': {'utrecht', 'utreg'},
            'sc heerenveen': {'heerenveen', 'sch'},
            'fc groningen': {'groningen', 'fcg'},
            'pec zwolle': {'zwolle', 'pec'},
            'vitesse arnhem': {'vitesse', 'arnhem'},
            'heracles almelo': {'heracles'},
            'willem ii tilburg': {'willem ii'},
            'fortuna sittard': {'fortuna'},
            'sparta rotterdam': {'sparta'},
            'rkc waalwijk': {'rkc'},
            'fc volendam': {'volendam'},
            'nec nijmegen': {'nec'},
            'go ahead eagles': {'eagles'},
            
            # Primeira Liga
            'fc porto': {'porto', 'fcp', 'dragões'},
            'sporting cp': {'sporting', 'scp', 'leões'},
            'sl benfica': {'benfica', 'slb', 'águias'},
            'sc braga': {'braga', 'scb'},
            'vitoria sc': {'guimaraes', 'vitoria'},
            'boavista fc': {'boavista'},
            'fc famalicao': {'famalicao'},
            'cs maritimo': {'maritimo'},
            'cd santa clara': {'santa clara'},
            'fc arouca': {'arouca'},
            'gil vicente fc': {'gil vicente'},
            'fc vizela': {'vizela'},
            'cd tondela': {'tondela'},
            'moreirense fc': {'moreirense'},
            'belenenses sad': {'belenenses'},
            'fc pacos de ferreira': {'pacos ferreira'},
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize team name for matching."""
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove common punctuation
        name = re.sub(r'[.,-]', ' ', name)
        
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        
        # Remove trailing/leading spaces
        name = name.strip()
        
        return name
    
    def _get_core_name(self, name: str) -> str:
        """Extract core team name by removing common suffixes."""
        normalized = self._normalize_name(name)
        
        # Remove common football club suffixes
        suffixes = ['fc', 'cf', 'sc', 'ac', 'united fc', 'city fc', 'football club',
                   'club de futbol', 'club atletico', 'sporting club', 'athletic club']
        
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)].strip()
                break
                
        return normalized
    
    def _jaccard_similarity(self, a: str, b: str) -> float:
        """Calculate Jaccard similarity between two strings."""
        set_a = set(a.split())
        set_b = set(b.split())
        
        if not set_a and not set_b:
            return 1.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _levenshtein_ratio(self, a: str, b: str) -> float:
        """Calculate normalized Levenshtein distance."""
        return difflib.SequenceMatcher(None, a, b).ratio()
    
    def _substring_match(self, query: str, target: str) -> float:
        """Check if query is a meaningful substring of target."""
        query_norm = self._normalize_name(query)
        target_norm = self._normalize_name(target)
        
        # Direct substring match
        if query_norm in target_norm:
            return len(query_norm) / len(target_norm)
        
        # Word-level substring match
        query_words = set(query_norm.split())
        target_words = set(target_norm.split())
        
        if query_words.issubset(target_words):
            return len(query_words) / len(target_words)
        
        return 0.0
    
    def match_team(self, query: str, candidates: List[str], 
                   threshold: float = 0.6) -> Optional[TeamMatch]:
        """
        Enhanced team matching with multiple algorithms.
        
        Args:
            query: Team name to match
            candidates: List of potential matches
            threshold: Minimum similarity threshold
            
        Returns:
            TeamMatch object or None if no good match found
        """
        if not query or not candidates:
            return None
        
        query_norm = self._normalize_name(query)
        best_match = None
        best_score = 0.0
        
        # Check for exact matches first
        for candidate in candidates:
            if self._normalize_name(candidate) == query_norm:
                return TeamMatch(
                    matched_name=candidate,
                    similarity=1.0,
                    confidence='exact',
                    method='exact_match',
                    aliases_used=[]
                )
        
        # Check aliases
        for candidate in candidates:
            candidate_norm = self._normalize_name(candidate)
            if candidate_norm in self.team_aliases:
                aliases = self.team_aliases[candidate_norm]
                for alias in aliases:
                    if self._normalize_name(alias) == query_norm:
                        return TeamMatch(
                            matched_name=candidate,
                            similarity=0.95,
                            confidence='high',
                            method='alias_match', 
                            aliases_used=[alias]
                        )
        
        # Fuzzy matching with multiple algorithms
        for candidate in candidates:
            candidate_norm = self._normalize_name(candidate)
            
            # Algorithm 1: Sequence matcher (handles typos well)
            seq_score = self._levenshtein_ratio(query_norm, candidate_norm)
            
            # Algorithm 2: Jaccard similarity (handles word order)
            jaccard_score = self._jaccard_similarity(query_norm, candidate_norm)
            
            # Algorithm 3: Substring matching (handles abbreviations)
            substr_score = self._substring_match(query_norm, candidate_norm)
            
            # Algorithm 4: Core name matching (ignores FC/CF suffixes)
            query_core = self._get_core_name(query)
            candidate_core = self._get_core_name(candidate)
            core_score = self._levenshtein_ratio(query_core, candidate_core)
            
            # Weighted combination of scores
            combined_score = (
                seq_score * 0.3 +
                jaccard_score * 0.25 +
                substr_score * 0.2 + 
                core_score * 0.25
            )
            
            # Bonus for close matches
            if seq_score > 0.85 or core_score > 0.9:
                combined_score += 0.1
            
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                method_used = 'fuzzy_combined'
                
                if seq_score == combined_score:
                    method_used = 'sequence_match'
                elif jaccard_score > 0.8:
                    method_used = 'jaccard_match' 
                elif substr_score > 0.5:
                    method_used = 'substring_match'
                elif core_score > 0.85:
                    method_used = 'core_name_match'
                
                confidence = 'high' if combined_score >= 0.85 else 'medium' if combined_score >= 0.7 else 'low'
                
                best_match = TeamMatch(
                    matched_name=candidate,
                    similarity=combined_score,
                    confidence=confidence,
                    method=method_used,
                    aliases_used=[]
                )
        
        return best_match
    
    def get_suggestions(self, query: str, candidates: List[str], 
                       max_suggestions: int = 3) -> List[TeamMatch]:
        """Get multiple suggestions for ambiguous matches."""
        suggestions = []
        
        for candidate in candidates:
            match = self.match_team(query, [candidate], threshold=0.3)
            if match:
                suggestions.append(match)
        
        # Sort by similarity score
        suggestions.sort(key=lambda x: x.similarity, reverse=True)
        
        return suggestions[:max_suggestions]