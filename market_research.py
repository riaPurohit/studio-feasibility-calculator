"""
market_research.py – Location-aware market intelligence for studio feasibility.

Data pipeline (3-step FIPS-based approach):
  1. Convert city + state abbreviation → state FIPS code (static lookup)
  2. Query U.S. Census Bureau API to resolve place FIPS code
     (free, no key: https://api.census.gov/data/...)
  3. Construct Census Reporter geoid (16000US{state}{place}) and fetch
     demographics from Census Reporter (free, no key: censusreporter.org)

Fallback: built-in metro lookup tables when any API is unreachable.

Why FIPS instead of free-text search?
  Census Reporter's /geo/search endpoint uses fuzzy text matching which
  can fail on common city names, abbreviations, or CDPs.  The FIPS
  approach guarantees an exact, deterministic match because every
  incorporated place in the U.S. has a unique FIPS code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# ────────────────────────────────────────────────────────────────────
# Constants & timeouts
# ────────────────────────────────────────────────────────────────────

_CENSUS_API_BASE = "https://api.census.gov/data"
_CR_BASE = "https://api.censusreporter.org/1.0"
_TIMEOUT_FIPS = 12        # seconds – Census Bureau can be slow
_TIMEOUT_CR = 10           # seconds – Census Reporter
_MAX_RETRIES = 2           # retry count for transient failures
_RETRY_BACKOFF = 1.5       # seconds between retries
_SQ_METERS_PER_SQ_MILE = 2_589_988.11

# ACS table IDs for Census Reporter
_TABLE_INCOME  = "B19013"   # Median household income
_TABLE_POP     = "B01003"   # Total population
_TABLE_AGE_SEX = "B01001"   # Sex by Age (fitness-target demo 25-54)
_TABLE_MED_AGE = "B01002"   # Median Age by Sex

# ────────────────────────────────────────────────────────────────────
# State FIPS lookup (all 50 states + DC + territories)
# ────────────────────────────────────────────────────────────────────

_STATE_FIPS: Dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "PR": "72",
    "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def _get_state_fips(state_abbr: str) -> Optional[str]:
    """Convert 2-letter state abbreviation to FIPS code."""
    return _STATE_FIPS.get(state_abbr.strip().upper())


# ────────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────────

@dataclass
class MarketData:
    """Container for all market-research results."""
    city: str = ""
    state: str = ""
    geoid: str = ""

    # Data source tracking
    data_source: str = "estimated"          # "live" | "estimated"
    data_source_detail: str = ""            # e.g. "Census Reporter (ACS 5-Year)"

    # Demographics
    median_household_income: Optional[float] = None
    income_source: str = ""
    population: Optional[int] = None
    population_density: Optional[float] = None    # people per sq mi
    density_source: str = ""
    land_area_sq_miles: Optional[float] = None
    median_age: Optional[float] = None
    pct_age_25_54: Optional[float] = None         # fitness target demo %
    age_source: str = ""

    # Business intelligence
    estimated_rent_sqft_annual: Optional[float] = None
    rent_source: str = ""
    competition_count: int = 0
    competition_details: List[Dict[str, Any]] = field(default_factory=list)
    competition_source: str = ""
    suggested_arpm_low: Optional[float] = None
    suggested_arpm_mid: Optional[float] = None
    suggested_arpm_high: Optional[float] = None
    demand_potential: str = ""      # "High" | "Medium" | "Low"

    # Neighborhood recommendations
    neighborhood_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    raw: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return self.data_source == "live"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "state": self.state,
            "geoid": self.geoid,
            "data_source": self.data_source,
            "data_source_detail": self.data_source_detail,
            "median_household_income": self.median_household_income,
            "income_source": self.income_source,
            "population": self.population,
            "population_density": self.population_density,
            "density_source": self.density_source,
            "land_area_sq_miles": self.land_area_sq_miles,
            "median_age": self.median_age,
            "pct_age_25_54": self.pct_age_25_54,
            "age_source": self.age_source,
            "estimated_rent_sqft_annual": self.estimated_rent_sqft_annual,
            "rent_source": self.rent_source,
            "competition_count": self.competition_count,
            "competition_details": self.competition_details,
            "competition_source": self.competition_source,
            "suggested_arpm": {
                "low": self.suggested_arpm_low,
                "mid": self.suggested_arpm_mid,
                "high": self.suggested_arpm_high,
            },
            "demand_potential": self.demand_potential,
            "neighborhood_suggestions": self.neighborhood_suggestions,
            "warnings": self.warnings,
            "raw": self.raw,
        }


# ────────────────────────────────────────────────────────────────────
# HTTP helper with retries
# ────────────────────────────────────────────────────────────────────

def _get_with_retry(url: str, params: Optional[Dict] = None,
                    timeout: float = 10, retries: int = _MAX_RETRIES,
                    headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    """GET request with retry on transient failures."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
                continue
            return None
        except requests.exceptions.ConnectionError:
            if attempt < retries:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
                continue
            return None
        except requests.exceptions.HTTPError:
            return None
        except Exception:
            return None
    return None


# ────────────────────────────────────────────────────────────────────
# Step 1: City + State → FIPS code (via Census Bureau API)
# ────────────────────────────────────────────────────────────────────

def _resolve_place_fips(city: str, state_abbr: str) -> Optional[Dict[str, str]]:
    """
    Use the Census Bureau ACS API to find a city's place FIPS code.

    Queries:  GET https://api.census.gov/data/2022/acs/acs5
              ?get=NAME&for=place:*&in=state:{state_fips}

    Returns dict with keys: state_fips, place_fips, full_name, geoid
    or None on failure.
    """
    state_fips = _get_state_fips(state_abbr)
    if not state_fips:
        return None

    url = f"{_CENSUS_API_BASE}/2022/acs/acs5"
    params = {
        "get": "NAME",
        "for": "place:*",
        "in": f"state:{state_fips}",
    }

    resp = _get_with_retry(url, params=params, timeout=_TIMEOUT_FIPS)
    if resp is None:
        return None

    try:
        rows = resp.json()
        if len(rows) < 2:
            return None

        city_lower = city.strip().lower()

        # Pass 1: exact match on "city city" or "city town" pattern
        for row in rows[1:]:
            name = row[0].lower()
            name_parts = name.split(",")[0].strip()
            clean_name = name_parts
            for suffix in [" city", " town", " village", " cdp", " borough", " municipality"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[:-len(suffix)].strip()
                    break
            if clean_name == city_lower:
                place_fips = row[2]
                geoid = f"16000US{state_fips}{place_fips}"
                return {
                    "state_fips": state_fips,
                    "place_fips": place_fips,
                    "full_name": row[0],
                    "geoid": geoid,
                }

        # Pass 2: partial/fuzzy match
        for row in rows[1:]:
            name = row[0].lower()
            if city_lower in name.split(",")[0]:
                place_fips = row[2]
                geoid = f"16000US{state_fips}{place_fips}"
                return {
                    "state_fips": state_fips,
                    "place_fips": place_fips,
                    "full_name": row[0],
                    "geoid": geoid,
                }

        return None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# Step 2: FIPS geoid → Census Reporter data
# ────────────────────────────────────────────────────────────────────

def _cr_get_geo_metadata(geoid: str) -> Optional[Dict]:
    """Fetch land area from Census Reporter's TIGER endpoint."""
    url = f"{_CR_BASE}/geo/tiger2022/{geoid}"
    resp = _get_with_retry(url, timeout=_TIMEOUT_CR)
    if resp is None:
        return None
    try:
        data = resp.json()
        if "properties" in data:
            return data["properties"]
        return data
    except Exception:
        return None


def _cr_get_data(geoid: str, table_ids: List[str]) -> Optional[Dict]:
    """Fetch ACS data from Census Reporter."""
    url = f"{_CR_BASE}/data/show/latest"
    params = {
        "table_ids": ",".join(table_ids),
        "geo_ids": geoid,
    }
    resp = _get_with_retry(url, params=params, timeout=_TIMEOUT_CR)
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception:
        return None


# ── Extraction helpers ──────────────────────────────────────────────

def _extract_income(data: Dict, geoid: str) -> Optional[float]:
    try:
        return float(data["data"][geoid][_TABLE_INCOME]["estimate"]["B19013_001"])
    except (KeyError, TypeError, ValueError):
        return None


def _extract_population(data: Dict, geoid: str) -> Optional[int]:
    try:
        return int(data["data"][geoid][_TABLE_POP]["estimate"]["B01003_001"])
    except (KeyError, TypeError, ValueError):
        return None


def _extract_median_age(data: Dict, geoid: str) -> Optional[float]:
    try:
        return float(data["data"][geoid][_TABLE_MED_AGE]["estimate"]["B01002_001"])
    except (KeyError, TypeError, ValueError):
        return None


def _extract_pct_age_25_54(data: Dict, geoid: str) -> Optional[float]:
    """Percentage of population aged 25-54 (fitness target demographic)."""
    try:
        age = data["data"][geoid][_TABLE_AGE_SEX]["estimate"]
        total = age.get("B01001_001")
        if not total or total <= 0:
            return None
        # Male 25-54: cols 010-015, Female 25-54: cols 034-039
        cols = [f"B01001_{i:03d}" for i in range(10, 16)] + \
               [f"B01001_{i:03d}" for i in range(34, 40)]
        target = sum(age.get(c, 0) or 0 for c in cols)
        return round((target / total) * 100, 1)
    except (KeyError, TypeError):
        return None


# ── Combined fetch ──────────────────────────────────────────────────

def fetch_census_data(city: str, state: str) -> Dict[str, Any]:
    """
    FIPS-based Census data pipeline.  No API key needed.

    Flow:
      1. Resolve city → FIPS via Census Bureau API
      2. Use FIPS-derived geoid with Census Reporter for demographics
      3. Return structured dict or empty dict on failure

    Returns dict with: geoid, full_name, income, population, density,
        land_area_sq_miles, median_age, pct_age_25_54, source
    """
    result: Dict[str, Any] = {}

    # Step 1: Resolve FIPS
    fips = _resolve_place_fips(city, state)
    if not fips:
        return {}

    geoid = fips["geoid"]
    result["geoid"] = geoid
    result["full_name"] = fips.get("full_name", "")
    result["fips_state"] = fips["state_fips"]
    result["fips_place"] = fips["place_fips"]

    # Step 2: Get geo metadata (land area)
    geo_meta = _cr_get_geo_metadata(geoid)
    if geo_meta:
        result["raw_geo"] = geo_meta
        aland = geo_meta.get("aland")
        if aland and aland > 0:
            result["land_area_sq_miles"] = round(aland / _SQ_METERS_PER_SQ_MILE, 2)

    # Step 3: Get ACS tables
    data = _cr_get_data(geoid, [_TABLE_INCOME, _TABLE_POP, _TABLE_AGE_SEX, _TABLE_MED_AGE])
    if data:
        result["raw_data"] = data

        income = _extract_income(data, geoid)
        if income is not None:
            result["income"] = income

        pop = _extract_population(data, geoid)
        if pop is not None:
            result["population"] = pop

        med_age = _extract_median_age(data, geoid)
        if med_age is not None:
            result["median_age"] = med_age

        pct = _extract_pct_age_25_54(data, geoid)
        if pct is not None:
            result["pct_age_25_54"] = pct

        if pop and result.get("land_area_sq_miles") and result["land_area_sq_miles"] > 0:
            result["density"] = round(pop / result["land_area_sq_miles"], 1)

    result["source"] = "Live Census Data (ACS 5-Year via censusreporter.org)"
    return result


# ────────────────────────────────────────────────────────────────────
# Fallback lookup tables
# ────────────────────────────────────────────────────────────────────

_METRO_DATA: Dict[str, Dict] = {
    # Major US metros
    "new york": {"income": 76_000, "density": 28_000, "rent": 65, "pop": 8_300_000},
    "los angeles": {"income": 70_000, "density": 8_300, "rent": 48, "pop": 3_900_000},
    "chicago": {"income": 65_000, "density": 11_900, "rent": 35, "pop": 2_700_000},
    "houston": {"income": 55_000, "density": 3_600, "rent": 28, "pop": 2_300_000},
    "phoenix": {"income": 60_000, "density": 3_100, "rent": 26, "pop": 1_600_000},
    "seattle": {"income": 105_000, "density": 8_400, "rent": 42, "pop": 750_000},
    "san francisco": {"income": 120_000, "density": 17_200, "rent": 60, "pop": 870_000},
    "denver": {"income": 75_000, "density": 4_500, "rent": 32, "pop": 710_000},
    "austin": {"income": 72_000, "density": 3_100, "rent": 34, "pop": 980_000},
    "miami": {"income": 45_000, "density": 12_300, "rent": 45, "pop": 440_000},
    "boston": {"income": 80_000, "density": 13_300, "rent": 45, "pop": 680_000},
    "nashville": {"income": 62_000, "density": 1_400, "rent": 30, "pop": 690_000},
    "dallas": {"income": 55_000, "density": 3_900, "rent": 28, "pop": 1_300_000},
    "atlanta": {"income": 65_000, "density": 3_500, "rent": 28, "pop": 500_000},
    "portland": {"income": 73_000, "density": 4_800, "rent": 30, "pop": 650_000},
    "minneapolis": {"income": 66_000, "density": 7_100, "rent": 28, "pop": 430_000},
    "san diego": {"income": 80_000, "density": 4_300, "rent": 42, "pop": 1_400_000},
    "charlotte": {"income": 60_000, "density": 2_800, "rent": 26, "pop": 870_000},
    "washington": {"income": 90_000, "density": 11_000, "rent": 50, "pop": 690_000},
    "philadelphia": {"income": 50_000, "density": 11_600, "rent": 30, "pop": 1_600_000},
    "raleigh": {"income": 72_000, "density": 2_900, "rent": 26, "pop": 470_000},
    "salt lake city": {"income": 65_000, "density": 1_700, "rent": 24, "pop": 200_000},
    "tampa": {"income": 55_000, "density": 3_300, "rent": 28, "pop": 390_000},
    "columbus": {"income": 56_000, "density": 4_100, "rent": 22, "pop": 900_000},
    "indianapolis": {"income": 50_000, "density": 2_400, "rent": 20, "pop": 880_000},
    # Washington State – Eastside / Puget Sound
    "bellevue": {"income": 122_000, "density": 4_500, "rent": 45, "pop": 153_000, "median_age": 36.8, "pct_25_54": 44.2},
    "redmond": {"income": 128_000, "density": 4_100, "rent": 40, "pop": 78_000, "median_age": 35.5, "pct_25_54": 46.1},
    "kirkland": {"income": 110_000, "density": 4_600, "rent": 42, "pop": 93_000, "median_age": 37.2, "pct_25_54": 43.5},
    "bothell": {"income": 105_000, "density": 4_000, "rent": 36, "pop": 51_000, "median_age": 36.0, "pct_25_54": 43.0},
    "woodinville": {"income": 115_000, "density": 2_800, "rent": 34, "pop": 14_000, "median_age": 40.5, "pct_25_54": 39.0},
    "issaquah": {"income": 120_000, "density": 3_200, "rent": 38, "pop": 41_000, "median_age": 37.8, "pct_25_54": 42.5},
    "sammamish": {"income": 170_000, "density": 3_500, "rent": 36, "pop": 67_000, "median_age": 39.5, "pct_25_54": 40.0},
    "mercer island": {"income": 165_000, "density": 3_800, "rent": 44, "pop": 26_000, "median_age": 44.0, "pct_25_54": 35.0},
    "renton": {"income": 75_000, "density": 4_200, "rent": 32, "pop": 107_000, "median_age": 35.2, "pct_25_54": 44.0},
    "kent": {"income": 68_000, "density": 3_800, "rent": 28, "pop": 136_000, "median_age": 33.5, "pct_25_54": 42.0},
    "tacoma": {"income": 62_000, "density": 4_000, "rent": 26, "pop": 220_000, "median_age": 35.0, "pct_25_54": 41.0},
    "everett": {"income": 60_000, "density": 3_600, "rent": 28, "pop": 112_000, "median_age": 34.0, "pct_25_54": 42.5},
    "lynnwood": {"income": 62_000, "density": 5_200, "rent": 30, "pop": 42_000, "median_age": 35.5, "pct_25_54": 43.0},
    "olympia": {"income": 66_000, "density": 3_000, "rent": 24, "pop": 56_000, "median_age": 36.5, "pct_25_54": 39.5},
    "spokane": {"income": 50_000, "density": 3_400, "rent": 20, "pop": 230_000, "median_age": 35.0, "pct_25_54": 38.5},
    # Other commonly searched cities
    "scottsdale": {"income": 92_000, "density": 1_400, "rent": 36, "pop": 242_000},
    "san jose": {"income": 130_000, "density": 5_800, "rent": 48, "pop": 1_010_000},
    "irvine": {"income": 110_000, "density": 4_100, "rent": 48, "pop": 310_000},
    "plano": {"income": 95_000, "density": 3_800, "rent": 30, "pop": 290_000},
    "frisco": {"income": 120_000, "density": 3_200, "rent": 32, "pop": 220_000},
}

_US_DEFAULTS = {"income": 60_000, "density": 3_500, "rent": 26, "pop": 100_000}


def _normalize_city(city: str) -> str:
    return city.strip().lower().replace(".", "")


def _lookup_metro(city: str) -> Dict:
    key = _normalize_city(city)
    for metro, data in _METRO_DATA.items():
        if metro in key or key in metro:
            return {**data, "matched_metro": metro}
    return {**_US_DEFAULTS, "matched_metro": None}


# ────────────────────────────────────────────────────────────────────
# Nominatim geocoding (OpenStreetMap – free, no API key)
# ────────────────────────────────────────────────────────────────────

_NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
_NOMINATIM_HEADERS = {"User-Agent": "StudioFeasibilityCalc/1.0"}
_TIMEOUT_NOM = 10


def _geocode_city(city: str, state: str) -> Optional[Dict[str, float]]:
    """
    Convert city + state → lat/lon via Nominatim.
    Returns {"lat": float, "lon": float} or None.
    """
    url = f"{_NOMINATIM_BASE}/search"
    params = {
        "q": f"{city}, {state}, United States",
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    }
    resp = _get_with_retry(url, params=params, timeout=_TIMEOUT_NOM,
                           headers=_NOMINATIM_HEADERS)
    if resp is None:
        return None
    try:
        results = resp.json()
        if results:
            return {
                "lat": float(results[0]["lat"]),
                "lon": float(results[0]["lon"]),
            }
    except (KeyError, IndexError, ValueError):
        pass
    return None


# ────────────────────────────────────────────────────────────────────
# Overpass API – real competitor data (OpenStreetMap – free, no key)
# ────────────────────────────────────────────────────────────────────

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_TIMEOUT_OVERPASS = 25


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in miles."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch_competitors_osm(lat: float, lon: float,
                           radius_meters: int = 8047) -> Tuple[int, List[Dict], str]:
    """
    Query Overpass API for fitness-related POIs within a radius.

    Default radius: 8047m ≈ 5 miles.
    Returns (count, details_list, source_string).
    """
    # Overpass QL: search for gyms, fitness centres, yoga, dance, sports centres
    query = f"""
    [out:json][timeout:20];
    (
      node["leisure"="fitness_centre"](around:{radius_meters},{lat},{lon});
      way["leisure"="fitness_centre"](around:{radius_meters},{lat},{lon});
      node["leisure"="sports_centre"](around:{radius_meters},{lat},{lon});
      way["leisure"="sports_centre"](around:{radius_meters},{lat},{lon});
      node["sport"="yoga"](around:{radius_meters},{lat},{lon});
      way["sport"="yoga"](around:{radius_meters},{lat},{lon});
      node["sport"="fitness"](around:{radius_meters},{lat},{lon});
      way["sport"="fitness"](around:{radius_meters},{lat},{lon});
      node["amenity"="gym"](around:{radius_meters},{lat},{lon});
      way["amenity"="gym"](around:{radius_meters},{lat},{lon});
      node["leisure"="dance"](around:{radius_meters},{lat},{lon});
      way["leisure"="dance"](around:{radius_meters},{lat},{lon});
    );
    out center body;
    """
    resp = _get_with_retry(_OVERPASS_URL, params={"data": query},
                           timeout=_TIMEOUT_OVERPASS)
    if resp is None:
        return (0, [], "")
    try:
        data = resp.json()
        elements = data.get("elements", [])
    except Exception:
        return (0, [], "")

    # Deduplicate by name (some POIs appear as both node and way)
    seen_names = set()
    unique = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        # Get coordinates (nodes have lat/lon directly, ways have center)
        e_lat = el.get("lat") or el.get("center", {}).get("lat")
        e_lon = el.get("lon") or el.get("center", {}).get("lon")

        # Deduplicate by name+approximate location
        dedup_key = name.lower() if name else f"{e_lat:.4f},{e_lon:.4f}"
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        # Classify the type
        leisure = tags.get("leisure", "")
        sport = tags.get("sport", "")
        amenity = tags.get("amenity", "")
        if "yoga" in sport or "yoga" in name.lower():
            cat = "Yoga Studio"
        elif "dance" in leisure or "dance" in name.lower():
            cat = "Dance / Barre Studio"
        elif "fitness" in leisure or "fitness" in sport or "gym" in amenity:
            cat = "Gym / Fitness Studio"
        elif "sports" in leisure:
            cat = "Sports Center"
        else:
            cat = "Fitness"

        # Skip unnamed POIs — they clutter the list with no useful info
        if not name:
            continue

        detail = {
            "name": name,
            "category": cat,
        }
        if e_lat and e_lon:
            # Accurate distance using Haversine formula
            dist = _haversine_miles(lat, lon, e_lat, e_lon)
            detail["distance_miles"] = round(dist, 1)

        unique.append(detail)

    # Filter out results beyond the requested radius (in miles)
    radius_miles = radius_meters / 1609.34
    unique = [d for d in unique if d.get("distance_miles", 0) <= radius_miles + 0.1]

    # Sort by distance
    unique.sort(key=lambda x: x.get("distance_miles", 999))

    source = f"OpenStreetMap (Overpass API, {radius_miles:.0f}-mi radius)"
    return (len(unique), unique, source)


# ────────────────────────────────────────────────────────────────────
# Nominatim reverse search – real nearby neighborhoods/suburbs
# ────────────────────────────────────────────────────────────────────

def _fetch_nearby_places_nominatim(lat: float, lon: float,
                                   city: str) -> List[Dict[str, Any]]:
    """
    Use Nominatim structured search to find real neighborhoods and
    nearby suburbs around a location.
    Returns list of neighborhood dicts with name, type, and distance.
    """
    results = []
    seen = set()

    # Strategy: search several offsets around the center to discover
    # different neighborhoods/suburbs (Nominatim returns the place
    # at each coordinate)
    offsets = [
        (0, 0),           # center
        (0.03, 0),        # ~2mi north
        (-0.03, 0),       # ~2mi south
        (0, 0.04),        # ~2mi east
        (0, -0.04),       # ~2mi west
        (0.02, 0.02),     # NE
        (-0.02, -0.02),   # SW
        (0.02, -0.02),    # NW
        (-0.02, 0.02),    # SE
    ]

    for dlat, dlon in offsets:
        url = f"{_NOMINATIM_BASE}/reverse"
        params = {
            "lat": lat + dlat,
            "lon": lon + dlon,
            "format": "json",
            "zoom": 14,  # neighborhood-level detail
            "addressdetails": 1,
        }
        resp = _get_with_retry(url, params=params, timeout=_TIMEOUT_NOM,
                               headers=_NOMINATIM_HEADERS)
        if resp is None:
            continue

        try:
            data = resp.json()
            addr = data.get("address", {})
        except Exception:
            continue

        # Extract neighborhood-level names
        for key in ["neighbourhood", "suburb", "quarter", "city_district"]:
            name = addr.get(key, "").strip()
            if name and name.lower() not in seen and name.lower() != city.lower():
                seen.add(name.lower())
                # Calculate approximate distance from center
                r_lat = float(data.get("lat", lat))
                r_lon = float(data.get("lon", lon))
                dist = math.sqrt((r_lat - lat)**2 + (r_lon - lon)**2) * 69.0

                results.append({
                    "area": name,
                    "type": key.replace("_", " ").title(),
                    "distance_miles": round(dist, 1),
                })

        # Respect Nominatim rate limit (1 req/sec)
        time.sleep(1.1)

    # Sort by distance
    results.sort(key=lambda x: x["distance_miles"])
    return results


# ────────────────────────────────────────────────────────────────────
# Heuristic helpers (fallbacks when APIs are unavailable)
# ────────────────────────────────────────────────────────────────────

def _estimate_rent_from_income(income: float, density: float) -> float:
    base = 18 + (income - 50_000) / 50_000 * 20
    if density >= 10_000:
        base *= 1.40
    elif density >= 5_000:
        base *= 1.20
    elif density >= 2_000:
        base *= 1.05
    return round(max(15, min(80, base)), 0)


def _estimate_arpm(income: float) -> Tuple[float, float, float]:
    base = max(99, min(399, income * 0.0025))
    return (round(base * 0.80, 0), round(base, 0), round(base * 1.25, 0))


def _estimate_demand(income: float, density: float, competition: int) -> str:
    score = 0
    if income >= 75_000:
        score += 2
    elif income >= 55_000:
        score += 1
    if density >= 5_000:
        score += 2
    elif density >= 2_000:
        score += 1
    if competition < 5:
        score += 2
    elif competition < 15:
        score += 1
    return "High" if score >= 5 else "Medium" if score >= 3 else "Low"


def _estimate_competition(population: int, density: float, keywords: List[str]) -> Tuple[int, List[Dict], str]:
    per_capita = 8_000 if density > 5_000 else 12_000 if density > 2_000 else 18_000
    est_total = max(1, int(population / per_capita))
    radius_fraction = min(1.0, 20 / max(1, math.sqrt(population / max(1, density))))
    est_nearby = max(1, int(est_total * radius_fraction))
    details = [{"keyword": kw, "estimated_count": max(1, est_nearby // len(keywords))} for kw in keywords]
    return (est_nearby, details, "Heuristic (population & density based)")


def _suggest_neighborhoods(income: float, density: float, population: int,
                           city: str, state: str) -> List[Dict[str, Any]]:
    """
    Generate data-driven neighborhood suggestions for a fitness studio.

    Uses income, density, and population to score suitability.
    Returns a list of neighborhood profiles ranked by fitness potential.
    """
    suggestions = []

    # Compute a suitability score (0-100)
    inc_score = min(40, max(0, (income - 40_000) / 2_000))       # 0-40 pts
    den_score = min(30, max(0, (density - 1_000) / 500))          # 0-30 pts
    pop_score = min(30, max(0, math.log10(max(1, population)) * 5))  # 0-30 pts
    total = round(inc_score + den_score + pop_score, 1)

    # Downtown / Urban Core
    downtown_score = round(min(100, total * 1.15), 1)
    suggestions.append({
        "area": f"Downtown {city}",
        "score": downtown_score,
        "rationale": "High foot traffic, density supports walk-in potential",
        "income_match": "High" if income >= 75_000 else "Medium",
        "density_match": "High" if density >= 5_000 else "Medium" if density >= 2_000 else "Low",
    })

    # Suburban / residential corridors
    if population >= 50_000:
        suburban_score = round(min(100, total * 0.95), 1)
        suggestions.append({
            "area": f"Suburban corridor near {city}",
            "score": suburban_score,
            "rationale": "Family-oriented, lower rent, ample parking",
            "income_match": "High" if income >= 70_000 else "Medium",
            "density_match": "Medium",
        })

    # Mixed-use / emerging areas
    if income >= 60_000:
        mixed_score = round(min(100, total * 1.05), 1)
        suggestions.append({
            "area": f"Mixed-use district in {city}",
            "score": mixed_score,
            "rationale": "Growing retail + residential, young professionals nearby",
            "income_match": "Medium" if income < 80_000 else "High",
            "density_match": "High" if density >= 4_000 else "Medium",
        })

    # Sort by score descending
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return suggestions


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def research_location(
    city: str,
    state: str,
    mode: str = "no_key",
    live_data: bool = True,
    radius_miles: int = 5,
    keywords: Optional[List[str]] = None,
    api_keys: Optional[Dict[str, str]] = None,
) -> MarketData:
    """
    Main entry point.  Returns MarketData with demographics.

    Pipeline:
      1. If live_data=True, resolve FIPS → fetch Census Reporter
      2. On any failure, fall back to built-in metro lookup

    Args:
        city: City name (e.g. "Bellevue")
        state: 2-letter state abbreviation (e.g. "WA")
        live_data: If True, attempt to fetch live Census data.
                   If False, skip network calls and use built-in estimates.
    """
    keywords = keywords or [
        "fitness studio", "Pilates", "HIIT",
        "boutique fitness", "cycling studio", "yoga studio",
    ]
    api_keys = api_keys or {}
    md = MarketData(city=city, state=state)

    # ── Step 1: Try live Census data (FIPS-based) ────────────────
    cr_data: Dict[str, Any] = {}
    if live_data:
        cr_data = fetch_census_data(city, state)

    if cr_data and cr_data.get("income") is not None:
        # SUCCESS: Live data fetched
        md.data_source = "live"
        md.data_source_detail = "Live Census Data (ACS 5-Year via censusreporter.org)"

        md.geoid = cr_data.get("geoid", "")
        md.median_household_income = cr_data["income"]
        md.income_source = md.data_source_detail

        md.population = cr_data.get("population")
        md.population_density = cr_data.get("density")
        md.land_area_sq_miles = cr_data.get("land_area_sq_miles")
        md.density_source = md.data_source_detail

        md.median_age = cr_data.get("median_age")
        md.pct_age_25_54 = cr_data.get("pct_age_25_54")
        md.age_source = md.data_source_detail

        md.raw["census_data"] = {
            k: v for k, v in cr_data.items()
            if k not in ("raw_data", "raw_geo")
        }
    else:
        # FALLBACK: Use built-in estimates
        metro = _lookup_metro(city)
        matched = metro.get("matched_metro")

        md.data_source = "estimated"
        if matched:
            md.data_source_detail = f"Estimated Data (built-in: {matched})"
        else:
            md.data_source_detail = "Estimated Data (U.S. national averages)"

        md.median_household_income = metro["income"]
        md.population = metro.get("pop")
        md.population_density = metro["density"]
        md.income_source = md.data_source_detail
        md.density_source = md.data_source_detail

        md.median_age = metro.get("median_age", 38.5)
        md.pct_age_25_54 = metro.get("pct_25_54", 40.0)
        md.age_source = md.data_source_detail

        md.raw["metro_lookup"] = metro

        if live_data and cr_data == {}:
            md.warnings.append(
                f"Could not fetch live Census data for '{city}, {state}'. "
                "Using built-in estimates instead. This may happen due to "
                "network issues or an unrecognized city name."
            )
        elif not live_data:
            md.warnings.append(
                "Live data fetching is disabled. Using built-in estimates."
            )
        if matched is None:
            md.warnings.append(
                f"No built-in data for '{city}, {state}'. Using U.S. national "
                "averages. Consider providing rent and ARPM manually for accuracy."
            )

    # ── Step 2: Rent estimate ───────────────────────────────────
    income = md.median_household_income or 60_000
    density = md.population_density or 3_500

    metro = _lookup_metro(city)
    if metro.get("matched_metro"):
        md.estimated_rent_sqft_annual = metro["rent"]
        md.rent_source = f"Metro benchmark ({metro['matched_metro']})"
    else:
        md.estimated_rent_sqft_annual = _estimate_rent_from_income(income, density)
        md.rent_source = "Estimated from income & density"

    # ── Step 3: Geocode city for real data lookups ──────────────
    coords = None
    if live_data:
        coords = _geocode_city(city, state)

    # ── Step 4: Competition (real OpenStreetMap data) ─────────
    pop = md.population or 100_000
    radius_m = int(radius_miles * 1609.34)

    if coords and live_data:
        cnt, details, src = _fetch_competitors_osm(
            coords["lat"], coords["lon"], radius_meters=radius_m)
        if cnt > 0 or src:
            md.competition_count = cnt
            md.competition_details = details
            md.competition_source = src
        else:
            # Overpass failed or returned nothing – fall back to heuristic
            cnt, details, src = _estimate_competition(pop, density, keywords)
            md.competition_count = cnt
            md.competition_details = details
            md.competition_source = src + " (Overpass API unavailable)"
            md.warnings.append(
                "Could not fetch live competitor data from OpenStreetMap. "
                "Using heuristic estimates based on population and density."
            )
    else:
        cnt, details, src = _estimate_competition(pop, density, keywords)
        md.competition_count = cnt
        md.competition_details = details
        md.competition_source = src

    # ── Step 5: ARPM suggestion ─────────────────────────────────
    low, mid, high = _estimate_arpm(income)
    md.suggested_arpm_low = low
    md.suggested_arpm_mid = mid
    md.suggested_arpm_high = high

    # ── Step 6: Demand potential ────────────────────────────────
    md.demand_potential = _estimate_demand(income, density, md.competition_count)

    # ── Step 7: Neighborhood suggestions (real OSM data) ──────
    if coords and live_data:
        real_neighborhoods = _fetch_nearby_places_nominatim(
            coords["lat"], coords["lon"], city)
        if real_neighborhoods:
            # Enrich with fitness suitability scoring
            for nbr in real_neighborhoods:
                inc_score = min(40, max(0, (income - 40_000) / 2_000))
                den_score = min(30, max(0, (density - 1_000) / 500))
                pop_score = min(30, max(0, math.log10(max(1, pop)) * 5))
                base_score = round(inc_score + den_score + pop_score, 1)
                # Closer neighborhoods score slightly higher
                dist_bonus = max(0, 5 - nbr.get("distance_miles", 5))
                nbr["score"] = round(min(100, base_score + dist_bonus), 1)
                nbr["income_match"] = "High" if income >= 75_000 else "Medium" if income >= 55_000 else "Low"
                nbr["density_match"] = "High" if density >= 5_000 else "Medium" if density >= 2_000 else "Low"
                nbr["rationale"] = f"Real neighborhood ({nbr['type']}, {nbr['distance_miles']} mi from center)"
                nbr["source"] = "OpenStreetMap (Nominatim)"
            real_neighborhoods.sort(key=lambda x: x["score"], reverse=True)
            md.neighborhood_suggestions = real_neighborhoods
        else:
            md.neighborhood_suggestions = _suggest_neighborhoods(
                income, density, pop, city, state)
            md.warnings.append(
                "Could not fetch real neighborhood data. Using general suggestions."
            )
    else:
        md.neighborhood_suggestions = _suggest_neighborhoods(
            income, density, pop, city, state)

    return md
