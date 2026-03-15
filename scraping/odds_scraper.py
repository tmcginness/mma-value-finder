"""
Scrape historical moneyline odds from BestFightOdds.com.

Two strategies:
1. Numbered UFC events (UFC 283, etc.): direct search by event number
2. Fight Night events: look up headliner fighter's profile to find event URL

Fighter validation ensures we never accept odds from the wrong event.
"""

import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path


BASE_URL = "https://www.bestfightodds.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
REQUEST_DELAY = 2.0


def _request(url, params=None):
    """Make a rate-limited request with retry."""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            return resp
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None


def find_event_by_number(event_num: str) -> str | None:
    """Find a numbered UFC event (e.g., 'UFC 300') on BFO."""
    resp = _request(f"{BASE_URL}/search", params={"query": f"UFC {event_num}"})
    if not resp or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for link in soup.find_all("a", href=re.compile(r"/events/")):
        text = link.get_text(strip=True).lower()
        bfo_num = re.search(r"ufc\s+(\d+)", text)
        if bfo_num and bfo_num.group(1) == event_num:
            return link["href"]
    return None


def find_event_by_fighter(
    fighter_name: str, ufc_fighters_on_date: set[str]
) -> tuple[str | None, list[dict]]:
    """Find an event by looking up a fighter's profile on BFO.

    Returns (event_path, scraped_fights) or (None, []).
    """
    last_name = fighter_name.strip().split()[-1]
    resp = _request(f"{BASE_URL}/search", params={"query": fighter_name})
    if not resp or resp.status_code != 200:
        return None, []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find fighter profile links
    profile_url = None
    for link in soup.find_all("a", href=re.compile(r"/fighters/")):
        link_text = link.get_text(strip=True).lower()
        if last_name.lower() in link_text:
            profile_url = link["href"]
            break

    if not profile_url:
        return None, []

    time.sleep(REQUEST_DELAY)

    resp = _request(f"{BASE_URL}{profile_url}")
    if not resp or resp.status_code != 200:
        return None, []

    soup = BeautifulSoup(resp.text, "html.parser")
    event_links = soup.find_all("a", href=re.compile(r"/events/"))

    seen = set()
    candidates = []
    for link in event_links:
        href = link["href"]
        if href not in seen and "ufc" in href.lower():
            seen.add(href)
            candidates.append(href)

    for event_path in candidates[:5]:
        time.sleep(REQUEST_DELAY)
        fights = scrape_event_odds(event_path)
        if fights and _validate_event(fights, ufc_fighters_on_date, min_overlap=0.3):
            return event_path, fights

    return None, []


def scrape_event_odds(event_path: str) -> list[dict]:
    """Scrape moneyline odds from a BFO event page."""
    resp = _request(f"{BASE_URL}{event_path}")
    if not resp or resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="odds-table")
    if len(tables) < 2:
        return []

    table = tables[1]
    rows = table.find_all("tr")

    fights = []
    current_pair = []

    for row in rows:
        if "pr" in row.get("class", []):
            continue

        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        first_cell = cells[0]
        fighter_link = first_cell.find("a", href=re.compile(r"/fighters/"))
        if not fighter_link:
            continue

        name = fighter_link.get_text(strip=True)

        # Extract odds from all sportsbook columns, compute consensus
        odds_values = []
        for cell in cells[1:]:
            text = re.sub(r"[▲▼]", "", cell.get_text(strip=True)).strip()
            if text and text != "n/a":
                try:
                    odds_values.append(int(text))
                except ValueError:
                    pass

        consensus = int(round(pd.Series(odds_values).median())) if odds_values else None
        current_pair.append({"name": name, "consensus_line": consensus})

        if len(current_pair) == 2:
            a, b = current_pair
            if a["consensus_line"] is not None and b["consensus_line"] is not None:
                fights.append({
                    "fighter_a": a["name"],
                    "fighter_b": b["name"],
                    "fighter_a_line": a["consensus_line"],
                    "fighter_b_line": b["consensus_line"],
                })
            current_pair = []

    return fights


def _normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    return name


def _names_match(name_a: str, name_b: str) -> bool:
    """Fuzzy match two fighter names."""
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if a == b:
        return True

    # Last name match
    last_a = a.split()[-1] if a.split() else ""
    last_b = b.split()[-1] if b.split() else ""
    if last_a == last_b and len(last_a) > 2:
        return True

    # Substring match for long names
    if len(a) > 5 and len(b) > 5 and (a in b or b in a):
        return True

    return False


def _validate_event(
    scraped_fights: list[dict],
    ufc_fighters_on_date: set[str],
    min_overlap: float = 0.3,
) -> bool:
    """Validate scraped fighters match our UFCStats data."""
    if not scraped_fights or not ufc_fighters_on_date:
        return False

    bfo_fighters = set()
    for fight in scraped_fights:
        bfo_fighters.add(fight["fighter_a"].strip())
        bfo_fighters.add(fight["fighter_b"].strip())

    matched = 0
    for bfo_name in bfo_fighters:
        for ufc_name in ufc_fighters_on_date:
            if _names_match(bfo_name, ufc_name):
                matched += 1
                break

    ratio = matched / len(bfo_fighters) if bfo_fighters else 0
    return ratio >= min_overlap


def scrape_historical_odds(
    events_df: pd.DataFrame,
    output_path: str = "data/historical_lines.csv",
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Scrape odds for all UFC events from start_date onwards."""
    events_df = events_df.copy()
    events_df["date"] = pd.to_datetime(events_df["date"], format="mixed")

    cutoff = pd.to_datetime(start_date)
    today = pd.Timestamp.now().normalize()
    mask = (events_df["date"] >= cutoff) & (events_df["date"] < today)
    relevant = events_df[mask]

    unique_events = (
        relevant[["event", "date"]]
        .drop_duplicates("event")
        .sort_values("date")
    )

    # Build fighter lookup per event for validation
    fighters_by_event = {}
    headliners_by_event = {}
    for event_name, group in relevant.groupby("event"):
        fighters = set()
        for _, row in group.iterrows():
            fighters.add(str(row["fighter_a"]).strip().lower())
            fighters.add(str(row["fighter_b"]).strip().lower())
        fighters_by_event[event_name] = fighters

        # Use first fight's fighters as headliners (typically the main event)
        first_row = group.iloc[0]
        headliners_by_event[event_name] = (
            str(first_row["fighter_a"]).strip(),
            str(first_row["fighter_b"]).strip(),
        )

    print(f"Scraping odds for {len(unique_events)} events from {start_date}...")

    # Load existing data for incremental scraping
    all_results = []
    existing_events = set()
    if Path(output_path).exists():
        existing = pd.read_csv(output_path)
        all_results = existing.to_dict("records")
        existing_events = set(existing["event"].unique())
        print(f"Resuming: {len(existing_events)} events already scraped.")

    scraped = 0
    failed = 0

    for _, row in unique_events.iterrows():
        event_name = row["event"]
        event_date = row["date"]

        if event_name in existing_events:
            continue

        ufc_fighters = fighters_by_event.get(event_name, set())
        event_path = None

        # Strategy 1: numbered UFC event (e.g., "UFC 300: ...")
        num_match = re.search(r"UFC\s+(\d+)", event_name)
        if num_match:
            event_path = find_event_by_number(num_match.group(1))
            if event_path:
                time.sleep(REQUEST_DELAY)
                fights = scrape_event_odds(event_path)
                if fights and _validate_event(fights, ufc_fighters):
                    for fight in fights:
                        fight["event"] = event_name
                        fight["date"] = event_date.strftime("%Y-%m-%d")
                        all_results.append(fight)
                    scraped += 1
                    print(f"  [OK] {event_name}: {len(fights)} fights ({scraped} done)")
                    time.sleep(REQUEST_DELAY)

                    if scraped % 10 == 0:
                        pd.DataFrame(all_results).to_csv(output_path, index=False)
                    continue
                else:
                    event_path = None

        # Strategy 2: look up headliner fighter's profile
        if not event_path:
            headliner_a, headliner_b = headliners_by_event.get(
                event_name, ("", "")
            )
            for fighter in [headliner_b, headliner_a]:
                if not fighter:
                    continue
                time.sleep(REQUEST_DELAY)
                event_path, fights = find_event_by_fighter(fighter, ufc_fighters)
                if event_path and fights:
                    for fight in fights:
                        fight["event"] = event_name
                        fight["date"] = event_date.strftime("%Y-%m-%d")
                        all_results.append(fight)
                    scraped += 1
                    print(f"  [OK] {event_name}: {len(fights)} fights ({scraped} done)")

                    if scraped % 10 == 0:
                        pd.DataFrame(all_results).to_csv(output_path, index=False)
                    break

            if not event_path:
                print(f"  [FAIL] {event_name}")
                failed += 1

        time.sleep(REQUEST_DELAY)

    # Final save
    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\nDone! Scraped {scraped} events, {failed} failed.")
        print(f"Total fight lines: {len(df)}")
        print(f"Saved to {output_path}")
    else:
        print("\nNo odds data scraped.")

    return df
