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
from pathlib import Path
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

            # Win/Loss indicator — UFCStats uses a green flag with text "win"
            # Also check for green flag CSS class as a fallback
            flag = cols[0].select_one(".b-flag__text")
            flag_text = ""
            if flag:
                # Use .string or first text node to avoid child element text
                flag_text = flag.get_text(strip=True).lower()
            green_flag = cols[0].select_one("a.b-flag_style_green")

            # Method, round, time — clean up embedded whitespace
            method = " ".join(cols[7].text.split()) if len(cols) > 7 else ""
            rnd = cols[8].text.strip() if len(cols) > 8 else ""
            fight_time = cols[9].text.strip() if len(cols) > 9 else ""

            # Weight class
            weight_class = cols[6].text.strip() if len(cols) > 6 else ""

            # Determine winner
            method_upper = method.split("\n")[0].strip().upper()
            if flag_text.startswith("win") or green_flag is not None:
                winner = names[0]
            elif flag_text in ("draw", "nc") or method_upper in ("OVERTURNED", "CNC"):
                winner = "Draw/NC"
            else:
                # Fallback: UFCStats lists winner first for decided fights
                if method_upper and method_upper not in ("OVERTURNED", "CNC"):
                    winner = names[0]
                else:
                    winner = "Draw/NC"

            fights.append({
                "event": event_name,
                "date": event_date,
                "fight_url": fight_url,
                "fighter_a": names[0],
                "fighter_b": names[1],
                "winner": winner,
                "method": method,
                "round": rnd,
                "time": fight_time,
                "weight_class": weight_class,
            })

        return fights

    # ── Fight Detail → Per-Fighter Stats ─────────────────────────────

    @staticmethod
    def _parse_of(text: str) -> tuple[int, int]:
        """Parse 'X of Y' format into (landed, attempted)."""
        match = re.match(r"(\d+)\s+of\s+(\d+)", text.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    @staticmethod
    def _parse_pct(text: str) -> float:
        """Parse '64%' or '---' into a float (0-100)."""
        text = text.strip().replace("%", "")
        if text == "---":
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_ctrl(text: str) -> int:
        """Parse control time 'M:SS' into total seconds."""
        text = text.strip()
        if not text or text == "--":
            return 0
        match = re.match(r"(\d+):(\d+)", text)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        return 0

    def scrape_fight_detail(self, fight_url: str) -> Optional[dict]:
        """
        Scrape detailed per-fighter stats from a fight detail page.

        Returns dict with per-fighter totals:
          fighter_a, fighter_b, and all stats prefixed with a_ or b_
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

        # ── Table 1: Totals ──
        # Cols: Fighter, KD, Sig.Str, Sig.Str%, Total Str, TD, TD%, Sub.Att, Rev, Ctrl
        totals_row = tables[0].select("tr")[1] if len(tables[0].select("tr")) > 1 else None
        if totals_row:
            cols = totals_row.select("td")
            if len(cols) >= 10:
                def _cell(col):
                    return [t.strip() for t in col.stripped_strings]

                names = _cell(cols[0])
                if len(names) >= 2:
                    stats["fighter_a"] = names[0]
                    stats["fighter_b"] = names[1]

                kd = _cell(cols[1])
                stats["a_kd"] = int(kd[0]) if len(kd) >= 2 else 0
                stats["b_kd"] = int(kd[1]) if len(kd) >= 2 else 0

                sig = _cell(cols[2])
                stats["a_sig_str_landed"], stats["a_sig_str_att"] = self._parse_of(sig[0]) if len(sig) >= 2 else (0, 0)
                stats["b_sig_str_landed"], stats["b_sig_str_att"] = self._parse_of(sig[1]) if len(sig) >= 2 else (0, 0)

                total = _cell(cols[4])
                stats["a_total_str_landed"], stats["a_total_str_att"] = self._parse_of(total[0]) if len(total) >= 2 else (0, 0)
                stats["b_total_str_landed"], stats["b_total_str_att"] = self._parse_of(total[1]) if len(total) >= 2 else (0, 0)

                td = _cell(cols[5])
                stats["a_td_landed"], stats["a_td_att"] = self._parse_of(td[0]) if len(td) >= 2 else (0, 0)
                stats["b_td_landed"], stats["b_td_att"] = self._parse_of(td[1]) if len(td) >= 2 else (0, 0)

                sub = _cell(cols[7])
                stats["a_sub_att"] = int(sub[0]) if len(sub) >= 2 else 0
                stats["b_sub_att"] = int(sub[1]) if len(sub) >= 2 else 0

                rev = _cell(cols[8])
                stats["a_rev"] = int(rev[0]) if len(rev) >= 2 else 0
                stats["b_rev"] = int(rev[1]) if len(rev) >= 2 else 0

                ctrl = _cell(cols[9])
                stats["a_ctrl_sec"] = self._parse_ctrl(ctrl[0]) if len(ctrl) >= 2 else 0
                stats["b_ctrl_sec"] = self._parse_ctrl(ctrl[1]) if len(ctrl) >= 2 else 0

        # ── Table 2: Sig. Strikes Breakdown ──
        # Cols: Fighter, Sig.Str, Sig.Str%, Head, Body, Leg, Distance, Clinch, Ground
        if len(tables) > 1:
            ss_row = tables[1].select("tr")[1] if len(tables[1].select("tr")) > 1 else None
            if ss_row:
                cols = ss_row.select("td")
                if len(cols) >= 9:
                    def _cell(col):
                        return [t.strip() for t in col.stripped_strings]

                    head = _cell(cols[3])
                    stats["a_head_landed"], stats["a_head_att"] = self._parse_of(head[0]) if len(head) >= 2 else (0, 0)
                    stats["b_head_landed"], stats["b_head_att"] = self._parse_of(head[1]) if len(head) >= 2 else (0, 0)

                    body = _cell(cols[4])
                    stats["a_body_landed"], stats["a_body_att"] = self._parse_of(body[0]) if len(body) >= 2 else (0, 0)
                    stats["b_body_landed"], stats["b_body_att"] = self._parse_of(body[1]) if len(body) >= 2 else (0, 0)

                    leg = _cell(cols[5])
                    stats["a_leg_landed"], stats["a_leg_att"] = self._parse_of(leg[0]) if len(leg) >= 2 else (0, 0)
                    stats["b_leg_landed"], stats["b_leg_att"] = self._parse_of(leg[1]) if len(leg) >= 2 else (0, 0)

                    dist = _cell(cols[6])
                    stats["a_dist_landed"], stats["a_dist_att"] = self._parse_of(dist[0]) if len(dist) >= 2 else (0, 0)
                    stats["b_dist_landed"], stats["b_dist_att"] = self._parse_of(dist[1]) if len(dist) >= 2 else (0, 0)

                    clinch = _cell(cols[7])
                    stats["a_clinch_landed"], stats["a_clinch_att"] = self._parse_of(clinch[0]) if len(clinch) >= 2 else (0, 0)
                    stats["b_clinch_landed"], stats["b_clinch_att"] = self._parse_of(clinch[1]) if len(clinch) >= 2 else (0, 0)

                    ground = _cell(cols[8])
                    stats["a_ground_landed"], stats["a_ground_att"] = self._parse_of(ground[0]) if len(ground) >= 2 else (0, 0)
                    stats["b_ground_landed"], stats["b_ground_att"] = self._parse_of(ground[1]) if len(ground) >= 2 else (0, 0)

        return stats if "fighter_a" in stats else None

    def scrape_all_fight_details(
        self,
        fights_csv: str = RAW_FIGHTS_CSV,
        output_path: str = "data/fight_details.csv",
    ) -> pd.DataFrame:
        """Scrape detailed stats for all fights. Supports incremental resume."""
        fights = pd.read_csv(fights_csv)
        fight_urls = fights["fight_url"].dropna().unique().tolist()

        # Resume from existing data
        all_stats = []
        scraped_urls = set()
        if Path(output_path).exists():
            existing = pd.read_csv(output_path)
            if not existing.empty:
                all_stats = existing.to_dict("records")
                scraped_urls = set(existing.get("fight_url", []))
                print(f"Resuming: {len(scraped_urls)} fights already scraped")

        remaining = [u for u in fight_urls if u not in scraped_urls]
        print(f"Scraping details for {len(remaining)} fights "
              f"({len(scraped_urls)} already done)...")

        for i, url in enumerate(tqdm(remaining, desc="Fight details")):
            detail = self.scrape_fight_detail(url)
            if detail:
                detail["fight_url"] = url
                all_stats.append(detail)

            # Save checkpoint every 100 fights
            if (i + 1) % 100 == 0:
                df = pd.DataFrame(all_stats)
                df.to_csv(output_path, index=False)

        df = pd.DataFrame(all_stats)
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} fight details to {output_path}")

        return df

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
