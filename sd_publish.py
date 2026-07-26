import json
import time

import requests

import sd_bayesian_model as model

GIST_FILENAME = "sd_runoff_state.json"
MAX_HISTORY_POINTS = 500


class PublishError(Exception):
    pass


def fetch_gist_content(gist_id, github_token):
    url = f"https://api.github.com/gists/{gist_id}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    file_data = data.get("files", {}).get(GIST_FILENAME)
    if not file_data:
        return None
    try:
        return json.loads(file_data["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def publish_snapshot(gist_id, github_token, n_sims=None):
    snap = model.snapshot(n_sims=n_sims) if n_sims else model.snapshot()

    existing = fetch_gist_content(gist_id, github_token)
    history = (existing.get("history") if existing else None) or []

    history.append({
        "updatedAt": snap["updatedAt"],
        "pctIn": snap["statewide"]["pctIn"],
        "rhodenShareMedian": snap["statewide"]["rhodenShareMedian"],
        "marginMedian": (snap["statewide"]["rhodenShareMedian"]
                          - (100 - snap["statewide"]["rhodenShareMedian"])),
    })
    if len(history) > MAX_HISTORY_POINTS:
        history = history[-MAX_HISTORY_POINTS:]

    snap["history"] = history

    url = f"https://api.github.com/gists/{gist_id}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "files": {
            GIST_FILENAME: {
                "content": json.dumps(snap, indent=2)
            }
        }
    }

    resp = requests.patch(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise PublishError(f"Gist update failed ({resp.status_code}): {resp.text[:500]}")

    return snap


def create_gist(github_token, description="SD Gov Runoff live state"):
    url = "https://api.github.com/gists"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "description": description,
        "public": True,
        "files": {
            GIST_FILENAME: {"content": json.dumps({"status": "initializing"})}
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 201:
        raise PublishError(f"Gist creation failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    gist_id = data["id"]
    raw_url = data["files"][GIST_FILENAME]["raw_url"]
    print("Created gist:", gist_id)
    print("Raw URL (use this in the website):", raw_url)
    print("NOTE: the raw_url above is version-pinned. For a URL that always")
    print("serves the latest content, use:")
    print(f"  https://gist.githubusercontent.com/<your-username>/{gist_id}/raw/{GIST_FILENAME}")
    return gist_id


def run_publish_loop(gist_id, github_token, interval_seconds=120):
    while True:
        try:
            snap = publish_snapshot(gist_id, github_token)
            pct = snap["statewide"]["pctIn"]
            print(f"[{snap['updatedAt']}] Published. {pct:.1%} of vote in. "
                  f"P(Rhoden)={snap['statewide']['pRhoden']:.1f}%")
        except PublishError as e:
            print("Publish failed:", e)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    print("Import this module and call create_gist(token) once to set up,")
    print("then publish_snapshot(gist_id, token) or run_publish_loop(...) to publish.")
