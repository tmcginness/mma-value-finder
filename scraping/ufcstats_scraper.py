"""
Scraper for UFCStats.com — pulls historical fight results and per-fighter stats.

Usage:
    scraper = UFCStatsScraper()
    fights_df = scraper.scrape_all_events()
    fighters_df = scraper.scrape_all_fighters()

Note: UFCStats is a community resource. Be respectful with request rates.
"""

import time
import re
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config.settings import (
    UFCSTATS_BASE_URL,
    REQUEST_DELAY_SECONDS,
    RAW_FIGHTS_CSV,
    RAW_FIGHTERS_CSV,
)


class UFCStatsScraper:
    """Scrapes fight and fighter data from UFCStats.com."""

    def __init__(self, delay: float = REQUEST_DELAY_SECONDS):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MMA-Value-Finder/1.0 (hobby project)"
        })

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page and return parsed BeautifulSoup, or None on failure."""
        try:
            time.sleep(self.delay)
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            print(f"  [WARN] Failed to fetch {url}: {e}")
            return None

    # ── Event List ───────────────────────────────────────────────────

    def get_event_urls(self) -> list[str]:
        """Get URLs for all completed UFC events."""
        urls = []
        page = 1
        while True:
            soup = self._get_soup(f"{UFCSTATS_BASE_URL}?page={page}")
            if soup is None:
                break

            rows = soup.select("tr.b-statistics__table-row")
            found = 0
            for row in rows:
                link = row.select_one("a.b-link")
                if link and link.get("href"):
                    urls.append(link["href"].strip())
                    found += 1

            if found == 0:
                break

            page += 1

        print(f"Found {len(urls)} events")
        return urls

    # ── Event Detail → Fight Rows ────────────────────────────────────

    def scrape_event(self, event_url: str) -> list[dict]:
        """Scrape all fights from a single event page."""
        soup = self._get_soup(event_url)
        if soup is None:
            return []

        # Event metadata
        event_name_tag = soup.select_one("h2.b-content__title span")
        event_name = event_name_tag.text.strip() if event_name_tag else "Unknown"

        date_tag = soup.select_one("li.b-list__box-list-item:first-child")
        event_date = ""
        if date_tag:
            date_text = date_tag.text.replace("Date:", "").strip()
            event_date = date_text

        fights = []
        fight_rows = soup.select("tr.b-fight-details__table-row")[1:]  # skip header

        for row in fight_rows:
            cols = row.select("td")
            if len(cols) < 8:
                continue

            fight_url = row.get("data-link", "")

            # Fighter names (two per fight)
            names = [a.text.strip() for a in cols[1].select("a")]
            if len(names) < 2:
                continue

            # Win/Loss indicators
            results = [i.text.strip() for i in cols[0].select("i")]

            # Method, round, time
            method = cols[7].text.strip() if len(cols) > 7 else ""
            rnd = cols[8].text.strip() if len(cols) > 8 else ""
            fight_time = cols[9].text.strip() if len(cols) > 9 else ""

            # Weight class
            weight_class = cols[6].text.strip() if len(cols) > 6 else ""

            fights.append({
                "event": event_name,
                "date": event_date,
                "fight_url": fight_url,
                "fighter_a": names[0],
                "fighter_b": names[1],
                "winner": names[0] if results and results[0] == "W" else (
                    names[1] if len(results) > 1 and results[1] == "W" else "Draw/NC"
                ),
                "method": method,
                "round": rnd,
                "time": fight_time,
                "weight_class": weight_class,
            })

        return fights

    # ── Fight Detail → Per-Fighter Stats ─────────────────────────────

    def scrape_fight_detail(self, fight_url: str) -> Optional[dict]:
        """
        Scrape detailed per-fighter stats from a fight detail page.
        Returns dict with totals and significant strikes for both fighters.

        This is where the rich data lives — strikes landed/attempted,
        takedowns, submissions, control time, etc.
        """
        if not fight_url:
            return None

        soup = self._get_soup(fight_url)
        if soup is None:
            return None

        tables = soup.select("table.b-fight-details__table")
        if not tables:
            return None

        stats = {}

        # Totals table (first table, "Totals" section)
        totals_rows = tables[0].select("tr") if len(tables) > 0 else []
        for row in totals_rows[1:]:  # skip header
            cols = [td.text.strip() for td in row.select("td")]
            if len(cols) >= 9:
                # cols: fighter, KD, sig_str, sig_str%, total_str, TD, TD%, sub_att, ctrl
                stats["knockdowns"] = cols[1]
                stats["sig_strikes"] = cols[2]
                stats["sig_strike_pct"] = cols[3]
                stats["total_strikes"] = cols[4]
                stats["takedowns"] = cols[5]
                stats["takedown_pct"] = cols[6]
                stats["sub_attempts"] = cols[7]
                stats["control_time"] = cols[8]
                break  # Just grab totals row

        return stats

    # ── Full Scrape ──────────────────────────────────────────────────

    def scrape_all_events(self, save: bool = True, incremental: bool = True) -> pd.DataFrame:
        """Scrape all events and their fights. Returns DataFrame of all fights.

        If incremental=True, saves progress after every 25 events so data
        isn't lost if the process is interrupted. Also supports resuming
        from a partial CSV.
        """
        event_urls = self.get_event_urls()
        all_fights = []
        start_idx = 0

        # Resume from existing partial data if available
        if incremental and save:
            try:
                existing = pd.read_csv(RAW_FIGHTS_CSV)
                if not existing.empty:
                    scraped_events = set(existing["event"].unique())
                    # Find first event URL not yet scraped
                    for i, url in enumerate(event_urls):
                        # We'll re-check by scraping, but skip events we already have
                        pass
                    all_fights = existing.to_dict("records")
                    # Skip URLs whose events we already scraped
                    remaining_urls = []
                    for url in event_urls:
                        soup = None  # Don't pre-fetch; filter by checking after scrape
                        remaining_urls.append(url)
                    # Simpler approach: just check scraped event count
                    start_idx = len(scraped_events)
                    print(f"Resuming: found {len(all_fights)} fights from {len(scraped_events)} events, "
                          f"skipping to event {start_idx}/{len(event_urls)}")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                pass

        for i, url in enumerate(tqdm(event_urls[start_idx:], desc="Scraping events",
                                     initial=start_idx, total=len(event_urls))):
            fights = self.scrape_event(url)
            all_fights.extend(fights)

            # Save progress every 25 events
            if incremental and save and (i + 1) % 25 == 0:
                pd.DataFrame(all_fights).to_csv(RAW_FIGHTS_CSV, index=False)
                print(f"\n  [checkpoint] Saved {len(all_fights)} fights after {start_idx + i + 1} events")

        df = pd.DataFrame(all_fights)
        if save and not df.empty:
            df.to_csv(RAW_FIGHTS_CSV, index=False)
            print(f"Saved {len(df)} fights to {RAW_FIGHTS_CSV}")

        return df

    # ── Fighter Career Stats ─────────────────────────────────────────

    def scrape_fighter_page(self, fighter_url: str) -> Optional[dict]:
        """Scrape a fighter's career stats from their profile page."""
        soup = self._get_soup(fighter_url)
        if soup is None:
            return None

        info = {}

        # Name
        name_tag = soup.select_one("span.b-content__title-highlight")
        info["name"] = name_tag.text.strip() if name_tag else ""

        # Record
        record_tag = soup.select_one("span.b-content__title-record")
        if record_tag:
            info["record"] = record_tag.text.replace("Record:", "").strip()

        # Career stats box
        stat_items = soup.select("li.b-list__box-list-item")
        for item in stat_items:
            text = item.text.strip()
            if "SLpM" in text:
                info["slpm"] = self._extract_number(text)
            elif "Str. Acc" in text:
                info["strike_accuracy"] = self._extract_number(text)
            elif "SApM" in text:
                info["sapm"] = self._extract_number(text)
            elif "Str. Def" in text:
                info["strike_defense"] = self._extract_number(text)
            elif "TD Avg" in text:
                info["td_avg"] = self._extract_number(text)
            elif "TD Acc" in text:
                info["td_accuracy"] = self._extract_number(text)
            elif "TD Def" in text:
                info["td_defense"] = self._extract_number(text)
            elif "Sub. Avg" in text:
                info["sub_avg"] = self._extract_number(text)
            elif "Height" in text:
                info["height"] = text.split(":")[-1].strip()
            elif "Weight" in text:
                info["weight"] = text.split(":")[-1].strip()
            elif "Reach" in text:
                info["reach"] = self._extract_number(text)
            elif "STANCE" in text.upper():
                info["stance"] = text.split(":")[-1].strip()
            elif "DOB" in text.upper():
                info["dob"] = text.split(":")[-1].strip()

        return info

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """Pull the first number from a stat string like 'SLpM: 4.32'."""
        match = re.search(r"[\d.]+", text.split(":")[-1])
        return float(match.group()) if match else None


# ── CLI Usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    scraper = UFCStatsScraper()
    print("Scraping all UFC events...")
    df = scraper.scrape_all_events(save=True)
    print(f"Done. {len(df)} fights scraped.")
