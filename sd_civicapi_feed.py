"""
Feed from civicAPI (civicapi.org) for the 2026 South Dakota Republican
gubernatorial runoff (race ID 84571, confirmed via
https://www.civicapi.org/results/elections/84571).

Uses the same domain and endpoint pattern already confirmed working for
the Arizona feed tonight: https://civicapi.org/api/v2/race/<id>, a
public, no-auth-required JSON endpoint. This SD race's exact response
has not been directly inspected (my sandbox can't reach civicapi.org
directly), so the parsing logic mirrors the AZ feed's structure
(region_results keyed by county slug, each with a candidates list) but
should be spot-checked against the first real response, same as the
AZ feed was validated once real data existed.

Unlike the AZ model, SD's model has no Early/Day-Of bucket split, so
there's no classifier to route through here -- each county's current
cumulative total is just reported directly.
"""

import requests

import sd_bayesian_model as model

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RACE_ID = 84571
RACE_URL = f"https://civicapi.org/api/v2/race/{RACE_ID}"

CANDIDATE_TO_KEY = {
    "RHODEN": "Rhoden",
    "DOEDEN": "Doeden",
}

COUNTY_NAMES = set(model.COUNTIES.keys())


class CivicAPIError(Exception):
    pass


def fetch_race(race_id=RACE_ID, timeout=12):
    resp = requests.get(f"https://civicapi.org/api/v2/race/{race_id}", timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return resp.json()


def diagnose_structure(data):
    print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  '{k}': list of {len(v)} items")
                if v:
                    print(f"    first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else v[0]}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with keys {list(v.keys())}")
            else:
                print(f"  '{k}': {v!r}")


def _candidate_key(name):
    name_upper = (name or "").upper()
    for surname, key in CANDIDATE_TO_KEY.items():
        if surname in name_upper:
            return key
    return None


REPORTING_PCT_KEYS = [
    "percent_reporting", "pct_reporting", "reporting_pct", "reporting",
    "pct_in", "percent_in", "percentReporting", "pctReporting",
]


def _extract_reporting_pct(entry):
    for key in REPORTING_PCT_KEYS:
        val = entry.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        # normalize: some APIs give 0-1, others 0-100
        return val / 100 if val > 1 else val
    return None


def find_county_breakdown(data):
    region_results = data.get("region_results")
    if not isinstance(region_results, dict):
        return None

    county_totals = {}
    for slug, entry in region_results.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or slug.replace("_", " ").title()
        name_clean = name.strip()

        matched_name = None
        for known in COUNTY_NAMES:
            if known.lower() == name_clean.lower():
                matched_name = known
                break
        if matched_name is None:
            continue

        totals = {"Rhoden": 0, "Doeden": 0}
        for c in entry.get("candidates", []):
            key = _candidate_key(c.get("name", ""))
            if key is not None:
                totals[key] += c.get("votes", 0) or 0
        totals["_pct_reporting"] = _extract_reporting_pct(entry)
        county_totals[matched_name] = totals

    return county_totals if county_totals else None


def get_statewide_totals(data):
    totals = {"Rhoden": 0, "Doeden": 0}
    for c in data.get("candidates", []):
        key = _candidate_key(c.get("name", ""))
        if key is not None:
            totals[key] += c.get("votes", 0) or 0
    return totals


def update_model_from_civicapi(skip_counties=None):
    if skip_counties is None:
        skip_counties = set()

    data = fetch_race()
    county_breakdown = find_county_breakdown(data)

    if county_breakdown is None:
        raise CivicAPIError(
            "civicAPI response has no recognizable county-level breakdown — "
            "only a statewide total is available. Call diagnose_structure() "
            "on the raw response to inspect it."
        )

    updated = []
    for county_name, totals in county_breakdown.items():
        if county_name in skip_counties:
            continue
        if county_name not in model.COUNTIES:
            continue

        county = model.COUNTIES[county_name]
        county.report(totals["Rhoden"], totals["Doeden"])

        pct_reporting = totals.get("_pct_reporting")
        if pct_reporting and pct_reporting > 0:
            reported_total = totals["Rhoden"] + totals["Doeden"]
            county.total_proj = reported_total / pct_reporting
            county.total_proj_is_measured = True

        updated.append(county_name)

    return updated


if __name__ == "__main__":
    data = fetch_race()
    diagnose_structure(data)
    print()
    breakdown = find_county_breakdown(data)
    if breakdown:
        print("County breakdown found:")
        print(breakdown)
    else:
        print("No county breakdown found — statewide only:")
        print(get_statewide_totals(data))
