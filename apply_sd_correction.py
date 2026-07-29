"""
ONE-TIME correction.
Resets every county's reported vote data to a clean slate, then applies
ONLY the real, current data given below. Any county not in this list
reverts to its pure pre-election baseline (no real data assumed).

This fixes counties that had been showing more "vote in" than they
actually have -- likely from an earlier, now-stale automated update
that overstated some counties' reported totals.

Run this ONCE, locally, with GIST_ID and GIST_TOKEN set as environment
variables (same ones Render uses):

    GIST_ID=your_gist_id GIST_TOKEN=your_token python3 apply_sd_correction.py
"""
import os
import sd_bayesian_model as model
import sd_publish as publish

GIST_ID = os.environ["GIST_ID"]
GIST_TOKEN = os.environ["GIST_TOKEN"]

# (Rhoden votes, Doeden votes), derived from the real margin+total data given.
# Counties not listed here are left with reported_rd = None (pre-election only).
REAL_DATA = {
    "Brown": (2608, 1091), "Brookings": (2542, 917), "Davison": (1928, 660),
    "Codington": (1248, 672), "Yankton": (1267, 505), "Lincoln": (1022, 527),
    "Hutchinson": (997, 369), "Union": (559, 544), "Tripp": (792, 286),
    "Clay": (690, 268), "Turner": (597, 256), "Charles Mix": (601, 184),
    "Day": (539, 236), "Spink": (521, 208), "Bon Homme": (507, 218),
    "McPherson": (497, 224), "Moody": (460, 243), "Brule": (477, 200),
    "Roberts": (430, 221), "Lake": (487, 149), "McCook": (403, 173),
    "Marshall": (411, 137), "Deuel": (333, 208), "Hanson": (356, 167),
    "Sully": (337, 134), "Campbell": (244, 120), "Lyman": (279, 79),
    "Miner": (244, 99), "Faulk": (231, 95), "Jerauld": (254, 72),
    "Hyde": (256, 43), "Kingsbury": (200, 85), "Gregory": (192, 75),
    "Mellette": (137, 48), "Haakon": (92, 27), "Jones": (66, 14),
    "Buffalo": (48, 12),
}

# Reset every county to a clean slate first
for county in model.COUNTIES.values():
    county.reported_rd = None

# Apply only the real, current data
for name, (r, d) in REAL_DATA.items():
    model.COUNTIES[name].report(r, d)
    print(f"{name}: Rhoden {r}, Doeden {d}")

print(f"\n{len(REAL_DATA)} counties set with real data, "
      f"{len(model.COUNTIES) - len(REAL_DATA)} reset to pre-election baseline.")

snap = publish.publish_snapshot(GIST_ID, GIST_TOKEN)
print(f"\nPublished corrected snapshot. {snap['statewide']['pctIn']:.1%} of vote in, "
      f"P(Rhoden)={snap['statewide']['pRhoden']:.1f}%")
