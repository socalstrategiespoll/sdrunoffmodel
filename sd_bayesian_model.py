"""
South Dakota Republican Gubernatorial Runoff — Live Election Model
Rhoden vs. Doeden, July 2026 runoff.

Architecture notes (differences from the AZ model this was adapted from):
- SD county results comingle early and Election-Day votes into a single
  reported total per county — there is no Early/Day-Of/Late-Mail split.
  Every county therefore has exactly ONE reporting bucket.
- Only two candidates are on this runoff ballot, so there is no
  "Other"-share instability to guard against the way AZ needed a
  decoupled Other-share fix. Vote share is modeled as a single
  Rhoden-vs-Doeden log-odds contrast.
- The "assume a big early swing reverts to baseline until there's
  real evidence" behavior the county-level modeling is built around
  falls directly out of the same hierarchical shrinkage math used in
  the AZ model: a county's own observed residual is blended against
  the regional/statewide prior, weighted by how much sampling
  uncertainty remains in that county's own count. Low reporting % ->
  wide uncertainty -> the prior dominates (assume reversion). High
  reporting % -> narrow uncertainty -> the county's own real number
  dominates (trust the swing). No separate "reversion mechanism" was
  needed; this behavior is inherent to the pooling model already.
- Regions: Minnehaha (Sioux Falls) and Pennington (Rapid City) are by
  far the two largest counties, similar to how Maricopa/Pima dominated
  the AZ map, so each gets its own single-county region. Every other
  county is grouped into East River or West River (split by the
  Missouri River), the standard, well-established SD political/
  cultural divide. This regional assignment is an assumption and can
  be revised.
"""

import numpy as np
from datetime import datetime, timezone

RNG = np.random.default_rng()

N_SIMS = 20000
CREDIBILITY_EXPONENT = 2.0
OUTLIER_LAMBDA = 3.0

TAU_FLOOR_STATE = 0.31
TAU_FLOOR_REGION = 0.08
TAU_FLOOR_COUNTY = 0.10
TAU2_CAP = 0.05

TAU_FLOOR_TURNOUT_STATE = 0.04
TAU_FLOOR_TURNOUT_REGION = 0.03
TAU_FLOOR_TURNOUT_COUNTY = 0.06

REGIONS = ["Minnehaha", "Pennington", "EastRiver", "WestRiver"]

COUNTY_REGION = {
    "Aurora": "EastRiver",
    "Beadle": "EastRiver",
    "Bennett": "WestRiver",
    "Bon Homme": "EastRiver",
    "Brookings": "EastRiver",
    "Brown": "EastRiver",
    "Brule": "EastRiver",
    "Buffalo": "EastRiver",
    "Butte": "WestRiver",
    "Campbell": "EastRiver",
    "Charles Mix": "EastRiver",
    "Clark": "EastRiver",
    "Clay": "EastRiver",
    "Codington": "EastRiver",
    "Corson": "WestRiver",
    "Custer": "WestRiver",
    "Davison": "EastRiver",
    "Day": "EastRiver",
    "Deuel": "EastRiver",
    "Dewey": "WestRiver",
    "Douglas": "EastRiver",
    "Edmunds": "EastRiver",
    "Fall River": "WestRiver",
    "Faulk": "EastRiver",
    "Grant": "EastRiver",
    "Gregory": "EastRiver",
    "Haakon": "WestRiver",
    "Hamlin": "EastRiver",
    "Hand": "EastRiver",
    "Hanson": "EastRiver",
    "Harding": "WestRiver",
    "Hughes": "WestRiver",
    "Hutchinson": "EastRiver",
    "Hyde": "EastRiver",
    "Jackson": "WestRiver",
    "Jerauld": "EastRiver",
    "Jones": "WestRiver",
    "Kingsbury": "EastRiver",
    "Lake": "EastRiver",
    "Lawrence": "WestRiver",
    "Lincoln": "EastRiver",
    "Lyman": "WestRiver",
    "Marshall": "EastRiver",
    "McCook": "EastRiver",
    "McPherson": "EastRiver",
    "Meade": "WestRiver",
    "Mellette": "WestRiver",
    "Miner": "EastRiver",
    "Minnehaha": "Minnehaha",
    "Moody": "EastRiver",
    "Oglala Lakota": "WestRiver",
    "Pennington": "Pennington",
    "Perkins": "WestRiver",
    "Potter": "EastRiver",
    "Roberts": "EastRiver",
    "Sanborn": "EastRiver",
    "Spink": "EastRiver",
    "Stanley": "WestRiver",
    "Sully": "EastRiver",
    "Todd": "WestRiver",
    "Tripp": "WestRiver",
    "Turner": "EastRiver",
    "Union": "EastRiver",
    "Walworth": "EastRiver",
    "Yankton": "EastRiver",
    "Ziebach": "WestRiver",
}

_CONFIG = {
    "Aurora": dict(total=426, prior_rhoden_share=0.575256),
    "Beadle": dict(total=1865, prior_rhoden_share=0.636895),
    "Bennett": dict(total=321, prior_rhoden_share=0.582872),
    "Bon Homme": dict(total=942, prior_rhoden_share=0.545231),
    "Brookings": dict(total=3566, prior_rhoden_share=0.6554),
    "Brown": dict(total=8191, prior_rhoden_share=0.605402),
    "Brule": dict(total=698, prior_rhoden_share=0.624657),
    "Buffalo": dict(total=62, prior_rhoden_share=0.6712),
    "Butte": dict(total=2104, prior_rhoden_share=0.624085),
    "Campbell": dict(total=375, prior_rhoden_share=0.571894),
    "Charles Mix": dict(total=1095, prior_rhoden_share=0.64823),
    "Clark": dict(total=893, prior_rhoden_share=0.586493),
    "Clay": dict(total=988, prior_rhoden_share=0.654384),
    "Codington": dict(total=3581, prior_rhoden_share=0.586539),
    "Corson": dict(total=251, prior_rhoden_share=0.622803),
    "Custer": dict(total=2545, prior_rhoden_share=0.493912),
    "Davison": dict(total=2668, prior_rhoden_share=0.676208),
    "Day": dict(total=872, prior_rhoden_share=0.639406),
    "Deuel": dict(total=740, prior_rhoden_share=0.54622),
    "Dewey": dict(total=301, prior_rhoden_share=0.661557),
    "Douglas": dict(total=1004, prior_rhoden_share=0.661452),
    "Edmunds": dict(total=889, prior_rhoden_share=0.561976),
    "Fall River": dict(total=1730, prior_rhoden_share=0.473357),
    "Faulk": dict(total=618, prior_rhoden_share=0.667403),
    "Grant": dict(total=1192, prior_rhoden_share=0.55588),
    "Gregory": dict(total=781, prior_rhoden_share=0.636644),
    "Haakon": dict(total=578, prior_rhoden_share=0.659837),
    "Hamlin": dict(total=1173, prior_rhoden_share=0.609528),
    "Hand": dict(total=772, prior_rhoden_share=0.702093),
    "Hanson": dict(total=539, prior_rhoden_share=0.563476),
    "Harding": dict(total=431, prior_rhoden_share=0.658321),
    "Hughes": dict(total=3942, prior_rhoden_share=0.782166),
    "Hutchinson": dict(total=1408, prior_rhoden_share=0.616392),
    "Hyde": dict(total=308, prior_rhoden_share=0.673117),
    "Jackson": dict(total=419, prior_rhoden_share=0.648261),
    "Jerauld": dict(total=336, prior_rhoden_share=0.654354),
    "Jones": dict(total=270, prior_rhoden_share=0.682988),
    "Kingsbury": dict(total=1155, prior_rhoden_share=0.616205),
    "Lake": dict(total=1832, prior_rhoden_share=0.644027),
    "Lawrence": dict(total=4900, prior_rhoden_share=0.633875),
    "Lincoln": dict(total=9712, prior_rhoden_share=0.65085),
    "Lyman": dict(total=521, prior_rhoden_share=0.627008),
    "Marshall": dict(total=602, prior_rhoden_share=0.664552),
    "McCook": dict(total=1013, prior_rhoden_share=0.581486),
    "McPherson": dict(total=743, prior_rhoden_share=0.696711),
    "Meade": dict(total=5260, prior_rhoden_share=0.626527),
    "Mellette": dict(total=191, prior_rhoden_share=0.649617),
    "Miner": dict(total=354, prior_rhoden_share=0.637836),
    "Minnehaha": dict(total=21473, prior_rhoden_share=0.627603),
    "Moody": dict(total=832, prior_rhoden_share=0.587962),
    "Oglala Lakota": dict(total=327, prior_rhoden_share=0.627079),
    "Pennington": dict(total=15567, prior_rhoden_share=0.598787),
    "Perkins": dict(total=792, prior_rhoden_share=0.672649),
    "Potter": dict(total=773, prior_rhoden_share=0.677011),
    "Roberts": dict(total=766, prior_rhoden_share=0.510792),
    "Sanborn": dict(total=337, prior_rhoden_share=0.650622),
    "Spink": dict(total=1114, prior_rhoden_share=0.617905),
    "Stanley": dict(total=1023, prior_rhoden_share=0.724547),
    "Sully": dict(total=486, prior_rhoden_share=0.64824),
    "Todd": dict(total=69, prior_rhoden_share=0.484509),
    "Tripp": dict(total=1156, prior_rhoden_share=0.636861),
    "Turner": dict(total=1691, prior_rhoden_share=0.585254),
    "Union": dict(total=1472, prior_rhoden_share=0.60139),
    "Walworth": dict(total=1024, prior_rhoden_share=0.647687),
    "Yankton": dict(total=2853, prior_rhoden_share=0.647334),
    "Ziebach": dict(total=161, prior_rhoden_share=0.712355),
}


def contrast_logodds(p):
    return np.log(p / (1 - p))


class County:
    MIN_REMAINING_FRACTION = 0.15  # total_proj auto-raises so reported never exceeds ~85% of it

    def __init__(self, name, total, prior_rhoden_share):
        self.name = name
        self.region = COUNTY_REGION[name]
        self.total_proj = total
        self.prior_rhoden_share = prior_rhoden_share
        self.reported_rd = None  # (rhoden, doeden) or None
        self.total_proj_is_measured = False  # True once set from a direct source (e.g. civicAPI's own percent_reporting)

    def report(self, rhoden, doeden):
        self.reported_rd = (rhoden, doeden)
        reported_total = rhoden + doeden
        floor_total = reported_total / (1 - self.MIN_REMAINING_FRACTION)
        if floor_total > self.total_proj and not self.total_proj_is_measured:
            self.total_proj = floor_total

    def reported_total(self):
        if self.reported_rd is None:
            return 0
        return self.reported_rd[0] + self.reported_rd[1]

    def observed_contrast(self):
        if self.reported_rd is None:
            return None
        r, d = self.reported_rd
        n = r + d
        if n == 0:
            return None
        obs = np.log((r + 0.5) / (d + 0.5))
        prior = contrast_logodds(self.prior_rhoden_share)
        var = 1 / (r + 0.5) + 1 / (d + 0.5)
        return obs - prior, var, n

    def observed_turnout(self):
        if self.reported_rd is None or self.total_proj <= 0:
            return None
        n = self.reported_total()
        if n == 0:
            return None
        shift = np.log(n / self.total_proj)
        var = 1.0 / max(n, 1)
        return shift, var, n


COUNTIES = {name: County(name, **cfg) for name, cfg in _CONFIG.items()}


def current_reporting_fraction():
    total_reported = sum(c.reported_total() for c in COUNTIES.values())
    total_proj = sum(c.total_proj for c in COUNTIES.values())
    return total_reported / total_proj if total_proj else 0.0


def hierarchical_pool(obs_dict, floor_state, floor_region, floor_county):
    """DerSimonian-Laird random-effects pooling, three levels:
    statewide, regional, county-idiosyncratic. obs_dict maps county
    name -> (shift, var, n)."""
    if not obs_dict:
        return None

    names = list(obs_dict.keys())
    shifts = np.array([obs_dict[n][0] for n in names])
    variances = np.array([obs_dict[n][1] for n in names])
    weights = 1.0 / variances

    state_mean_fixed = np.sum(weights * shifts) / np.sum(weights)
    if len(names) >= 2:
        q = np.sum(weights * (shifts - state_mean_fixed) ** 2)
        df = len(names) - 1
        c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        tau2 = min(tau2, TAU2_CAP)
    else:
        tau2 = floor_county ** 2

    re_weights = 1.0 / (variances + tau2)
    state_mean = np.sum(re_weights * shifts) / np.sum(re_weights)
    state_var = 1.0 / np.sum(re_weights)

    region_means, region_vars = {}, {}
    for region in REGIONS:
        region_names = [n for n in names if COUNTIES[n].region == region]
        if not region_names:
            region_means[region] = 0.0
            region_vars[region] = floor_region ** 2
            continue
        r_shifts = np.array([obs_dict[n][0] for n in region_names])
        r_vars = np.array([obs_dict[n][1] for n in region_names])
        r_weights = 1.0 / (r_vars + tau2)
        r_mean = np.sum(r_weights * (r_shifts - state_mean)) / np.sum(r_weights)
        r_var = 1.0 / np.sum(r_weights)
        region_means[region] = r_mean
        region_vars[region] = r_var

    n_reporting = len(names)
    return dict(state_mean=state_mean, state_var=state_var,
                region_means=region_means, region_vars=region_vars,
                county_tau2=tau2, n_reporting=n_reporting)


def coverage():
    reporting = [c for c in COUNTIES.values() if c.reported_rd is not None and c.total_proj > 0]
    if not reporting:
        return 0.0

    breadth_pct = len(reporting) / len(COUNTIES)
    avg_completeness = sum(min(c.reported_total() / c.total_proj, 1.0) for c in reporting) / len(reporting)

    return (breadth_pct * avg_completeness) ** CREDIBILITY_EXPONENT


def empty_factor(floor_state, floor_region, floor_county):
    return dict(state_mean=0.0, state_var=floor_state ** 2,
                region_means={r: 0.0 for r in REGIONS},
                region_vars={r: floor_region ** 2 for r in REGIONS},
                county_tau2=floor_county ** 2, n_reporting=0)


def get_rd_factor():
    obs = {}
    for name, c in COUNTIES.items():
        r = c.observed_contrast()
        if r is not None:
            obs[name] = r
    pooled = hierarchical_pool(obs, TAU_FLOOR_STATE, TAU_FLOOR_REGION, TAU_FLOOR_COUNTY)
    if pooled is None:
        return empty_factor(TAU_FLOOR_STATE, TAU_FLOOR_REGION, TAU_FLOOR_COUNTY)
    cred = coverage()
    pooled["state_mean"] *= cred
    pooled["region_means"] = {g: v * cred for g, v in pooled["region_means"].items()}
    return pooled


def get_turnout_factor():
    obs = {}
    for name, c in COUNTIES.items():
        r = c.observed_turnout()
        if r is not None:
            obs[name] = r
    pooled = hierarchical_pool(obs, TAU_FLOOR_TURNOUT_STATE, TAU_FLOOR_TURNOUT_REGION,
                                TAU_FLOOR_TURNOUT_COUNTY)
    if pooled is None:
        return empty_factor(TAU_FLOOR_TURNOUT_STATE, TAU_FLOOR_TURNOUT_REGION,
                             TAU_FLOOR_TURNOUT_COUNTY)
    cred = coverage()
    pooled["state_mean"] *= cred
    pooled["region_means"] = {g: v * cred for g, v in pooled["region_means"].items()}
    return pooled


def county_specific_effect(county, factor, quantity):
    """Blend a county's own observed residual against the population
    prior (tau2), weighted by how much sampling uncertainty remains in
    that county's own count. This is the mechanism that makes a big
    early swing get treated with skepticism (wide variance -> lean on
    the regional/state prior) until enough of that same county's own
    vote is in to trust it (narrow variance -> trust the county's own
    number). No separate 'reversion' rule is needed beyond this."""
    tau2 = factor["county_tau2"]

    if quantity == "T":
        obs = county.observed_turnout()
    else:
        obs = county.observed_contrast()

    if obs is None:
        return 0.0, tau2

    shift_c, var_c, n = obs
    residual = shift_c - factor["state_mean"] - factor["region_means"][county.region]
    posterior_var = 1 / (1 / tau2 + 1 / var_c)
    posterior_mean = residual * (posterior_var / var_c)
    return posterior_mean, posterior_var


def draw_factor(factor, n_sims):
    state = RNG.normal(factor["state_mean"], np.sqrt(factor["state_var"]), n_sims)
    region_draws = {}
    for region in REGIONS:
        region_draws[region] = RNG.normal(
            factor["region_means"][region], np.sqrt(factor["region_vars"][region]), n_sims
        )
    tau = np.sqrt(factor["county_tau2"])
    return state, region_draws, tau


def county_point_estimate_remaining(name):
    county = COUNTIES[name]
    rd_factor = get_rd_factor()

    reported_n = county.reported_total()

    if county.total_proj_is_measured:
        adjusted_total = county.total_proj
    else:
        t_factor = get_turnout_factor()
        shift_T = t_factor["state_mean"] + t_factor["region_means"][county.region]
        post_mean_T, _ = county_specific_effect(county, t_factor, "T")
        shift_T += post_mean_T
        adjusted_total = county.total_proj * np.exp(shift_T)

    remaining = max(0.0, adjusted_total - reported_n)
    if remaining <= 0:
        return {"Rhoden": 0.0, "Doeden": 0.0}

    prior_logodds = contrast_logodds(county.prior_rhoden_share)
    shift_RD = rd_factor["state_mean"] + rd_factor["region_means"][county.region]
    post_mean_RD, _ = county_specific_effect(county, rd_factor, "RD")
    shift_RD += post_mean_RD

    rd_ratio = np.exp(prior_logodds + shift_RD)
    rhoden_share = rd_ratio / (1 + rd_ratio)

    return {"Rhoden": remaining * rhoden_share, "Doeden": remaining * (1 - rhoden_share)}


def simulate(n_sims=N_SIMS):
    t_factor = get_turnout_factor()
    rd_factor = get_rd_factor()

    state_T, region_T, tau_T = draw_factor(t_factor, n_sims)
    state_RD, region_RD, tau_RD = draw_factor(rd_factor, n_sims)

    final_rhoden = np.zeros(n_sims)
    final_doeden = np.zeros(n_sims)

    for c in COUNTIES.values():
        reported_n = c.reported_total()
        if c.reported_rd:
            final_rhoden += c.reported_rd[0]
            final_doeden += c.reported_rd[1]

        post_mean_T, post_var_T = county_specific_effect(c, t_factor, "T")
        county_noise_T = RNG.normal(post_mean_T, np.sqrt(post_var_T), n_sims)
        turnout_shift = state_T + region_T[c.region] + county_noise_T
        adjusted_total = c.total_proj * np.exp(turnout_shift)
        remaining = np.maximum(0.0, adjusted_total - reported_n)

        prior_logodds = contrast_logodds(c.prior_rhoden_share)
        post_mean_RD, post_var_RD = county_specific_effect(c, rd_factor, "RD")
        county_noise_RD = RNG.normal(post_mean_RD, np.sqrt(post_var_RD), n_sims)
        shift_RD = state_RD + region_RD[c.region] + county_noise_RD

        sampling_sd = 1 / np.sqrt(np.maximum(remaining, 1))
        sampling_noise = RNG.normal(0, sampling_sd, n_sims)

        logodds_RD = prior_logodds + shift_RD + sampling_noise
        rd_ratio = np.exp(logodds_RD)
        rhoden_share = rd_ratio / (1 + rd_ratio)

        final_rhoden += remaining * rhoden_share
        final_doeden += remaining * (1 - rhoden_share)

    total = final_rhoden + final_doeden
    return final_rhoden, final_doeden, total


def report_status():
    rhoden, doeden, total = simulate()
    rhoden_share = rhoden / total
    win_prob = float(np.mean(rhoden > doeden) * 100)
    print("=======================================================")
    print("SD REPUBLICAN GOVERNOR RUNOFF - LIVE PROJECTION")
    print("=======================================================")
    print(f"P(Rhoden wins):   {win_prob:.2f}%")
    print(f"Rhoden share:     median {np.median(rhoden_share)*100:.2f}%  "
          f"[{np.percentile(rhoden_share,25)*100:.2f}, {np.percentile(rhoden_share,75)*100:.2f}]")
    print(f"Total votes:      median {np.median(total):,.0f}  "
          f"[{np.percentile(total,25):,.0f}, {np.percentile(total,75):,.0f}]")
    margin = rhoden - doeden
    print(f"Rhoden-Doeden margin: median {np.median(margin):,.0f} votes  "
          f"[{np.percentile(margin,25):,.0f}, {np.percentile(margin,75):,.0f}]")


def snapshot(n_sims=N_SIMS):
    counties_out = {}
    for name, county in COUNTIES.items():
        reported = county.reported_rd or (0, 0)
        remaining = county_point_estimate_remaining(name)
        counties_out[name] = {
            "reported": {"Rhoden": reported[0], "Doeden": reported[1],
                         "total": reported[0] + reported[1]},
            "remaining": {"Rhoden": remaining["Rhoden"], "Doeden": remaining["Doeden"],
                          "total": remaining["Rhoden"] + remaining["Doeden"]},
        }

    rhoden, doeden, total = simulate(n_sims)
    rhoden_share = rhoden / total
    win_prob = float(np.mean(rhoden > doeden) * 100)

    reported_total = sum(c["reported"]["total"] for c in counties_out.values())
    reported_rhoden = sum(c["reported"]["Rhoden"] for c in counties_out.values())
    reported_doeden = sum(c["reported"]["Doeden"] for c in counties_out.values())
    projected_total = sum(c.total_proj for c in COUNTIES.values())

    statewide_out = {
        "pRhoden": win_prob,
        "rhodenShareMedian": float(np.median(rhoden_share)) * 100,
        "rhodenShareP25": float(np.percentile(rhoden_share, 25)) * 100,
        "rhodenShareP75": float(np.percentile(rhoden_share, 75)) * 100,
        "totalMedian": float(np.median(total)),
        "reportedTotal": reported_total,
        "reportedRhoden": reported_rhoden,
        "reportedDoeden": reported_doeden,
        "reportedRhodenShare": reported_rhoden / reported_total if reported_total else 0,
        "projectedTotal": projected_total,
        "pctIn": reported_total / projected_total if projected_total else 0,
    }

    return {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counties": counties_out,
        "statewide": statewide_out,
    }


if __name__ == "__main__":
    report_status()
