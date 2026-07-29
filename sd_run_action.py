import os
import sys

import sd_bayesian_model as model
import sd_civicapi_feed as civicapi
import sd_publish as publish

GIST_ID = os.environ.get("GIST_ID")
GIST_TOKEN = os.environ.get("GIST_TOKEN")


def main():
    if not GIST_ID or not GIST_TOKEN:
        print("Missing GIST_ID or GIST_TOKEN environment variables.")
        sys.exit(1)

    try:
        updated = civicapi.update_model_from_civicapi()
        print(f"[civicAPI] updated OK ({len(updated)}/66 counties)")
    except Exception as e:
        print("[civicAPI] FAILED:", e)

    try:
        snap = publish.publish_snapshot(GIST_ID, GIST_TOKEN)
        pct = snap["statewide"]["pctIn"]
        pRhoden = snap["statewide"]["pRhoden"]
        print(f"[Publish] OK — {pct:.1%} of vote in, P(Rhoden)={pRhoden:.1f}%")
    except Exception as e:
        print("[Publish] FAILED:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
