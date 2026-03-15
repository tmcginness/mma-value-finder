"""
Scrape fighter records from Tapology.com to supplement UFC newcomers.

Tapology has complete pro MMA records including DWCS, regional promotions,
and international organizations — exactly what we need for fighters with
fewer than 2 UFC fights.

Usage:
    scraper = TapologyScraper()
    records = scraper.scrape_fighters(["Hecher Sosa", "Bolaji Oki"])
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config.settings import (
    TAPOLOGY_BASE_URL,
    TAPOLOGY_DELAY_SECONDS,
    TAPOLOGY_RECORDS_CSV,
)


class TapologyScraper:
    """Scrapes fighter records from Tapology.com."""

    def __init__(self, delay: float = TAPOLOGY_DELAY_SECONDS):
        self.delay = delay

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page via curl and return parsed BeautifulSoup.

        Tapology blocks Python's requests library, so we use curl which
        handles TLS fingerprinting correctly.
        """
        time.sleep(self.delay)
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [
                        "curl", "-s", "-L",
                        "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                        "-H", "Accept: text/html,application/xhtml+xml",
                        "-H", "Accept-Language: en-US,en;q=0.9",
                        "--max-time", "15",
                        url,
                    ],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(f"curl failed (rc={result.returncode})")
                return BeautifulSoup(result.stdout, "lxml")
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  [WARN] Request failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [WARN] Failed to fetch {url}: {e}")
        return None

    # ── Search ────────────────────────────────────────────────────────

    def search_fighter(self, name: str) -> Optional[str]:
        """
        Search Tapology for a fighter and return their profile URL path.
        Returns None if no match found.
        """
        soup = self._get_soup(
            f"{TAPOLOGY_BASE_URL}/search?term={quote(name)}"
            "&mainSearchFilter=fighters"
        )
        if soup is None:
            return None

        # Search results are in a table inside div.searchResultsFighter
        results_div = soup.select_one("div.searchResultsFighter")
        if not results_div:
            return None

        # Find fighter links in search results
        links = results_div.select("a[href*='/fightcenter/fighters/']")
        if not links:
            return None

        # Try to match the name
        name_norm = _normalize_name(name)
        for link in links:
            link_name = _normalize_name(link.text)
            if _names_match(name_norm, link_name):
                return link["href"]

        # Fallback: return first result if only one
        if len(links) == 1:
            return links[0]["href"]

        return None

    # ── Fighter Record ────────────────────────────────────────────────

    def scrape_fighter_record(
        self, profile_path: str, fighter_name: str
    ) -> list[dict]:
        """
        Scrape a fighter's complete fight record from their Tapology profile.

        Returns a list of fight dicts sorted chronologically (oldest first).
        Only includes actual fights (skips cancelled bouts).
        """
        url = (
            f"{TAPOLOGY_BASE_URL}{profile_path}"
            if profile_path.startswith("/")
            else profile_path
        )
        soup = self._get_soup(url)
        if soup is None:
            return []

        results_section = soup.select_one("section.fighterFightResults")
        if not results_section:
            return []

        fight_blocks = results_section.select("div[data-bout-id]")
        fights = []

        for block in fight_blocks:
            fight = self._parse_fight_block(block, fighter_name)
            if fight is not None:
                fights.append(fight)

        # Sort oldest first
        fights.sort(key=lambda f: f["date"])
        return fights

    def _parse_fight_block(
        self, block, fighter_name: str
    ) -> Optional[dict]:
        """Parse a single fight block from the fighter's record page."""
        status = block.get("data-status", "")
        division = block.get("data-division", "")
        sport = block.get("data-sport", "")

        # Skip cancelled bouts and non-MMA
        if status == "cancelled" or sport != "mma":
            return None

        # Skip amateur fights
        if division != "pro":
            return None

        # Result (W/L/D/NC)
        result_div = block.select_one("div.result > div:first-child")
        result_letter = result_div.text.strip() if result_div else ""

        result = {
            "W": "win", "L": "loss", "D": "draw", "N": "nc",
        }.get(result_letter, None)
        if result is None:
            return None

        # Method abbreviation (KO, SUB, DEC, etc.)
        method_div = block.select_one("div.-rotate-90")
        method_abbrev = method_div.text.strip().upper() if method_div else ""

        # Opponent
        opp_link = block.select_one("a[href*='/fightcenter/fighters/']")
        opponent = opp_link.text.strip() if opp_link else "Unknown"

        # Full method detail from bout link (first one has the detail)
        bout_links = block.select("a[href*='/fightcenter/bouts/']")
        method_full = bout_links[0].text.strip() if bout_links else ""

        # Event name — find event link that doesn't contain just a date
        event_name = ""
        for el in block.select("a[href*='/fightcenter/events/']"):
            text = el.text.strip()
            # Skip links that are just dates (e.g. "2026\nMar 14")
            if text and not re.match(r"^\d{4}", text):
                event_name = text
                break

        # Date — scan all spans in block for year and month-day patterns
        year = ""
        month_day = ""
        for span in block.select("span"):
            t = span.text.strip()
            if re.match(r"^\d{4}$", t) and not year:
                year = t
            elif re.match(r"^[A-Z][a-z]{2}\s+\d{1,2}$", t) and not month_day:
                month_day = t

        date_str = ""
        if year and month_day:
            try:
                date_str = pd.to_datetime(
                    f"{month_day} {year}", format="%b %d %Y"
                ).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                date_str = f"{year}-01-01"
        elif year:
            date_str = f"{year}-01-01"

        if not date_str:
            return None

        # Parse method and round from method_full
        # Examples: "Decision · Unanimous", "Rear Naked Choke · 2:17 · R3",
        #           "KO/TKO · 3:12 · R1"
        method_clean = _clean_method(method_abbrev, method_full)
        round_num, fight_time = _parse_round_time(method_full)

        # Promotion from logo img
        promo_img = block.select_one("img[alt]")
        promotion = promo_img.get("alt", "") if promo_img else ""

        is_ufc = promotion.upper() in ("UFC", "DWCS") or "UFC" in event_name.upper()

        return {
            "fighter": fighter_name,
            "date": date_str,
            "opponent": opponent,
            "result": result,
            "method": method_clean,
            "round": round_num,
            "time": fight_time,
            "event": event_name,
            "promotion": promotion,
            "is_ufc": is_ufc,
        }

    # ── Batch Scrape ──────────────────────────────────────────────────

    def scrape_fighters(
        self,
        fighter_names: list[str],
        output_path: str = TAPOLOGY_RECORDS_CSV,
    ) -> pd.DataFrame:
        """
        Scrape Tapology records for a list of fighters.
        Supports incremental resume — skips fighters already in the CSV.
        """
        all_records = []
        scraped_fighters = set()

        # Resume from existing data
        if Path(output_path).exists():
            existing = pd.read_csv(output_path)
            if not existing.empty:
                all_records = existing.to_dict("records")
                scraped_fighters = set(existing["fighter"].unique())
                print(f"Resuming: {len(scraped_fighters)} fighters already scraped")

        remaining = [n for n in fighter_names if n not in scraped_fighters]
        if not remaining:
            print("All fighters already scraped.")
            return pd.DataFrame(all_records)

        print(f"Scraping Tapology records for {len(remaining)} fighters...")
        found = 0
        failed = 0

        for i, name in enumerate(tqdm(remaining, desc="Tapology")):
            profile_path = self.search_fighter(name)
            if not profile_path:
                print(f"  [MISS] {name}: not found on Tapology")
                failed += 1
                continue

            records = self.scrape_fighter_record(profile_path, name)
            if records:
                # Only keep non-UFC fights (we already have UFC data)
                non_ufc = [r for r in records if not r["is_ufc"]]
                all_records.extend(non_ufc)
                found += 1
                print(f"  [OK] {name}: {len(non_ufc)} pre-UFC fights "
                      f"({len(records)} total)")
            else:
                print(f"  [EMPTY] {name}: found profile but no records")
                failed += 1

            # Checkpoint every 25 fighters
            if (i + 1) % 25 == 0:
                df = pd.DataFrame(all_records)
                if not df.empty:
                    df.to_csv(output_path, index=False)

        # Final save
        df = pd.DataFrame(all_records)
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"\nDone! Found records for {found} fighters, "
                  f"{failed} not found.")
            print(f"Total pre-UFC fight records: {len(df)}")
            print(f"Saved to {output_path}")
        else:
            print("\nNo Tapology records scraped.")

        return df


# ── Helpers ───────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Normalize fighter name for matching."""
    # Remove nicknames in quotes
    name = re.sub(r'["\u201c\u201d].*?["\u201c\u201d]', "", name)
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    return " ".join(name.split())


def _names_match(a: str, b: str) -> bool:
    """Fuzzy match two normalized fighter names."""
    if a == b:
        return True

    # Last name match
    last_a = a.split()[-1] if a.split() else ""
    last_b = b.split()[-1] if b.split() else ""

    if last_a == last_b and len(last_a) > 2:
        # Also check first initial
        first_a = a.split()[0][0] if a.split() else ""
        first_b = b.split()[0][0] if b.split() else ""
        if first_a == first_b:
            return True

    # Substring match for longer names
    if len(a) > 5 and len(b) > 5 and (a in b or b in a):
        return True

    return False


def _clean_method(abbrev: str, full_method: str) -> str:
    """Normalize method string to match our data conventions."""
    abbrev = abbrev.upper()
    full_upper = full_method.upper()

    if abbrev in ("KO", "TKO") or "KO" in full_upper:
        return "ko_tko"
    if abbrev == "SUB" or "SUBMISSION" in full_upper or "CHOKE" in full_upper or "LOCK" in full_upper:
        return "submission"
    if abbrev == "DEC" or "DECISION" in full_upper:
        if "SPLIT" in full_upper:
            return "split_decision"
        if "MAJORITY" in full_upper:
            return "majority_decision"
        return "unanimous_decision"
    if "DQ" in abbrev or "DISQUALIFICATION" in full_upper:
        return "dq"
    if "NC" in abbrev or "NO CONTEST" in full_upper:
        return "nc"

    return "unknown"


def _parse_round_time(method_full: str) -> tuple[Optional[int], str]:
    """
    Extract round and time from method string.
    Examples:
        "Rear Naked Choke · 2:17 · R3" → (3, "2:17")
        "Decision · Unanimous" → (None, "")
        "KO/TKO · 3:12 · R1" → (1, "3:12")
    """
    round_match = re.search(r"R(\d+)", method_full)
    round_num = int(round_match.group(1)) if round_match else None

    time_match = re.search(r"(\d+:\d+)", method_full)
    fight_time = time_match.group(1) if time_match else ""

    return round_num, fight_time
