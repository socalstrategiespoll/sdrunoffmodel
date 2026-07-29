"""
Feed from South Dakota Secretary of State's live results site
(electionresults.sd.gov) for the July 28, 2026 gubernatorial runoff
(Rhoden vs. Doeden).

Each county has its own static, GET-able URL:
    https://electionresults.sd.gov/ResultsSW.aspx?type=CTYALL&cty=<code>&map=CTY

Confirmed by direct inspection on 2026-07-25: this URL pattern renders
the county's races (including "Governor") with candidate name, party,
and vote count directly in the page -- no login, no JS postback needed
for reading results (the "Export" links elsewhere on the site ARE
JS-postback-driven and were deliberately avoided in favor of this
simpler, directly-renders-the-numbers page).

NOT YET VALIDATED AGAINST LIVE DATA: this was built by inspecting the
page's structure while it was showing all-zero pre-election
placeholders. The parsing logic is written to be robust to exact
formatting (it scans text content rather than depending on specific
CSS classes/ids, since those weren't directly inspectable), but it has
not been tested against a page actually showing non-zero results. Test
this against real data as soon as any county reports, and adjust the
parser if the live structure differs from what's assumed here.
"""

import re
import requests
from bs4 import BeautifulSoup

import sd_bayesian_model as model

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

BASE_URL = "https://electionresults.sd.gov/ResultsSW.aspx"

# County name -> the site's internal county code, taken directly from
# electionresults.sd.gov/ResultsList.aspx?type=CTYALL
COUNTY_CODES = {
    "Aurora": "10", "Beadle": "04", "Bennett": "11", "Bon Homme": "12",
    "Brookings": "06", "Brown": "03", "Brule": "13", "Buffalo": "14",
    "Butte": "15", "Campbell": "16", "Charles Mix": "17", "Clark": "18",
    "Clay": "19", "Codington": "05", "Corson": "20", "Custer": "21",
    "Davison": "08", "Day": "22", "Deuel": "23", "Dewey": "24",
    "Douglas": "25", "Edmunds": "26", "Fall River": "27", "Faulk": "28",
    "Grant": "29", "Gregory": "30", "Haakon": "31", "Hamlin": "32",
    "Hand": "33", "Hanson": "34", "Harding": "35", "Hughes": "36",
    "Hutchinson": "37", "Hyde": "38", "Jackson": "39", "Jerauld": "40",
    "Jones": "41", "Kingsbury": "42", "Lake": "43", "Lawrence": "09",
    "Lincoln": "44", "Lyman": "45", "Marshall": "48", "McCook": "46",
    "McPherson": "47", "Meade": "49", "Mellette": "50", "Miner": "51",
    "Minnehaha": "01", "Moody": "52", "Oglala Lakota": "65",
    "Pennington": "02", "Perkins": "53", "Potter": "54", "Roberts": "55",
    "Sanborn": "56", "Spink": "57", "Stanley": "58", "Sully": "59",
    "Todd": "67", "Tripp": "60", "Turner": "61", "Union": "62",
    "Walworth": "63", "Yankton": "07", "Ziebach": "64",
}

# The two runoff candidates, as their names appear on the site
RHODEN_NAME = "Larry Rhoden"
DOEDEN_NAME = "Toby Doeden"

KNOWN_PARTIES = {"Republican", "Democratic", "Independent", "Libertarian", "Nonpartisan"}


class SDFeedError(Exception):
    pass


def county_url(county_name):
    code = COUNTY_CODES[county_name]
    return f"{BASE_URL}?type=CTYALL&cty={code}&map=CTY"


def fetch_county_page(county_name, timeout=15):
    url = county_url(county_name)
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_governor_results(html_text, county_name):
    """Parse the Governor race out of a county results page. Scans the
    page's text content (not specific CSS selectors, since the raw
    structure wasn't directly inspectable) for the 'Governor' section,
    then walks forward collecting (name, party, votes) triples until
    hitting 'total votes'."""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    try:
        gov_idx = lines.index("Governor")
    except ValueError:
        raise SDFeedError(f"No 'Governor' section found on {county_name}'s page")

    precincts_fully, precincts_total = None, None
    for j in range(gov_idx, min(gov_idx + 60, len(lines))):
        m = re.match(r"Precincts Fully:\s*([\d,]+)\s*/\s*([\d,]+)", lines[j])
        if m:
            precincts_fully = int(m.group(1).replace(",", ""))
            precincts_total = int(m.group(2).replace(",", ""))
            break

    candidates = {}
    j = gov_idx
    while j < len(lines) - 2:
        if lines[j].lower() == "total votes":
            break
        if lines[j + 1] in KNOWN_PARTIES and re.match(r"^[\d,]+$", lines[j + 2]):
            name = lines[j]
            votes = int(lines[j + 2].replace(",", ""))
            candidates[name] = votes
            j += 3
        else:
            j += 1

    if not candidates:
        raise SDFeedError(f"No candidates parsed for {county_name}'s Governor race")

    return {
        "candidates": candidates,
        "precincts_fully": precincts_fully,
        "precincts_total": precincts_total,
    }


def update_model_from_county(county_name, html_text=None):
    if html_text is None:
        html_text = fetch_county_page(county_name)

    parsed = parse_governor_results(html_text, county_name)
    candidates = parsed["candidates"]

    rhoden = candidates.get(RHODEN_NAME)
    doeden = candidates.get(DOEDEN_NAME)

    if rhoden is None or doeden is None:
        raise SDFeedError(
            f"{county_name}: expected candidates '{RHODEN_NAME}' and '{DOEDEN_NAME}', "
            f"found {list(candidates.keys())}"
        )

    county = model.COUNTIES[county_name]
    county.report(rhoden, doeden)

    precincts_fully = parsed["precincts_fully"]
    precincts_total = parsed["precincts_total"]
    if precincts_fully and precincts_total:
        pct_in = precincts_fully / precincts_total
        reported_total = rhoden + doeden
        if pct_in > 0:
            live_estimate = reported_total / pct_in
            county.total_proj = live_estimate

    return {"Rhoden": rhoden, "Doeden": doeden,
            "precincts_fully": precincts_fully,
            "precincts_total": precincts_total}


def update_all_counties():
    updated, failed = [], {}
    for county_name in COUNTY_CODES:
        try:
            update_model_from_county(county_name)
            updated.append(county_name)
        except Exception as e:
            failed[county_name] = str(e)
    return updated, failed
