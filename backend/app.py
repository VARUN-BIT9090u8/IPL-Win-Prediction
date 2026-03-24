from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd
import os
import tensorflow as tf
load_model = tf.keras.models.load_model
from rapidfuzz import process, fuzz
from flask import Flask, render_template


app = Flask(__name__)

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH        = os.path.join(BASE_DIR, "..", "model", "ipl_lstm_model.h5")
ENCODER_PATH      = os.path.join(BASE_DIR, "..", "model", "encoders.pkl")
SCALER_PATH       = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")
SCORE_MODEL_PATH  = os.path.join(BASE_DIR, "..", "model", "score_model.h5")
SCORE_SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "score_scaler.pkl")
SCORE_TEAM_ENC    = os.path.join(BASE_DIR, "..", "model", "score_team_encoder.pkl")
SCORE_VENUE_ENC   = os.path.join(BASE_DIR, "..", "model", "score_venue_encoder.pkl")
DATASET_PATH      = os.path.join(BASE_DIR, "..", "dataset", "IPL.csv")

# =========================================================
# LOAD DATA
# =========================================================
DF = pd.read_csv(DATASET_PATH, low_memory=False)
DF.columns = DF.columns.str.lower().str.strip()

# =========================================================
# NAME NORMALIZATION
# =========================================================
NAME_FIX = {
    "Royal Challengers Bengaluru":  "Royal Challengers Bangalore",
    "Royal Challengers Bangalore":  "Royal Challengers Bangalore",
    "Punjab Kings":                 "Kings XI Punjab",
    "Delhi Capitals":               "Delhi Daredevils",
    "Rising Pune Supergiant":       "Rising Pune Supergiants",
    "Chennai Super Kings":          "Chennai Super Kings",
    "Mumbai Indians":               "Mumbai Indians",
    "Kolkata Knight Riders":        "Kolkata Knight Riders",
    "Rajasthan Royals":             "Rajasthan Royals",
    "Sunrisers Hyderabad":          "Sunrisers Hyderabad",
    "Gujarat Titans":               "Gujarat Titans",
    "Lucknow Super Giants":         "Lucknow Super Giants",
    "Kings XI Punjab":              "Kings XI Punjab",
    "Delhi Daredevils":             "Delhi Daredevils",
    "Deccan Chargers":              "Deccan Chargers",
    "Kochi Tuskers Kerala":         "Kochi Tuskers Kerala",
    "Pune Warriors":                "Pune Warriors",
    "Rising Pune Supergiants":      "Rising Pune Supergiants",
}

def normalize(team: str) -> str:
    return NAME_FIX.get(team, team)

def get_team_variants(canonical: str) -> set:
    variants = {canonical}
    for k, v in NAME_FIX.items():
        if v == canonical:
            variants.add(k)
    return variants

# =========================================================
# VENUE NORMALIZATION — consolidate duplicate venue name variants
# =========================================================
VENUE_FIX = {
    "Arun Jaitley Stadium":                          "Arun Jaitley Stadium, Delhi",
    "Brabourne Stadium":                             "Brabourne Stadium, Mumbai",
    "Dr DY Patil Sports Academy":                    "Dr DY Patil Sports Academy, Mumbai",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Eden Gardens":                                  "Eden Gardens, Kolkata",
    "Himachal Pradesh Cricket Association Stadium":  "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "M Chinnaswamy Stadium":                         "M Chinnaswamy Stadium, Bengaluru",
    "M.Chinnaswamy Stadium":                         "M Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium":                        "MA Chidambaram Stadium, Chepauk, Chennai",
    "MA Chidambaram Stadium, Chepauk":               "MA Chidambaram Stadium, Chepauk, Chennai",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Maharashtra Cricket Association Stadium":       "Maharashtra Cricket Association Stadium, Pune",
    "Punjab Cricket Association IS Bindra Stadium":  "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Punjab Cricket Association Stadium, Mohali":    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
    "Rajiv Gandhi International Stadium":            "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal":     "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Sawai Mansingh Stadium":                        "Sawai Mansingh Stadium, Jaipur",
    "Wankhede Stadium":                              "Wankhede Stadium, Mumbai",
    "Zayed Cricket Stadium, Abu Dhabi":              "Sheikh Zayed Stadium",
}

def normalize_venue(venue: str) -> str:
    return VENUE_FIX.get(venue, venue)


# =========================================================
# NORMALIZE COLUMNS
# Actual CSV column for match winner = "match_won_by"
# =========================================================
_norm = lambda x: normalize(x) if isinstance(x, str) else x

for col in ["batting_team", "bowling_team", "toss_winner", "match_won_by"]:
    if col in DF.columns:
        DF[col] = DF[col].map(_norm)

# Create unified "winner" alias
DF["winner"] = DF["match_won_by"] if "match_won_by" in DF.columns else pd.NA

# Normalize venue column
if "venue" in DF.columns:
    DF["venue"] = DF["venue"].map(lambda x: normalize_venue(x) if isinstance(x, str) else x)

# =========================================================
# BUILD OPTIMIZED PLAYER SEARCH INDEX
# =========================================================
ALL_PLAYERS = sorted(
    set(
        pd.concat([DF["batter"], DF["bowler"]])
        .dropna()
        .str.strip()
        .unique()
        .tolist()
    )
)

PLAYER_LOWER_INDEX = {name.lower(): name for name in ALL_PLAYERS}
PLAYER_LOWER_LIST  = list(PLAYER_LOWER_INDEX.keys())

# =========================================================
# LOAD MODELS
# =========================================================
model = load_model(MODEL_PATH, compile=False ,safe_mode=False)
with open(ENCODER_PATH,  "rb") as f: encoders = pickle.load(f)
with open(SCALER_PATH,   "rb") as f: scaler   = pickle.load(f)

score_model = load_model(SCORE_MODEL_PATH, compile=False)
with open(SCORE_SCALER_PATH, "rb") as f: score_scaler        = pickle.load(f)
with open(SCORE_TEAM_ENC,    "rb") as f: score_team_encoder  = pickle.load(f)
with open(SCORE_VENUE_ENC,   "rb") as f: score_venue_encoder = pickle.load(f)

# =========================================================
# CONSTANTS
# =========================================================
TEAM_MAP = {
    "CSK":  "Chennai Super Kings",
    "MI":   "Mumbai Indians",
    "RCB":  "Royal Challengers Bangalore",
    "KKR":  "Kolkata Knight Riders",
    "RR":   "Rajasthan Royals",
    "SRH":  "Sunrisers Hyderabad",
    "DC":   "Delhi Daredevils",
    "GT":   "Gujarat Titans",
    "LSG":  "Lucknow Super Giants",
    "PBKS": "Kings XI Punjab",
}

TEAM_INFO = {
    "Chennai Super Kings":         {"titles":5,  "captain":"MS Dhoni",       "coach":"Stephen Fleming",    "home":"MA Chidambaram Stadium",   "founded":2008},
    "Mumbai Indians":              {"titles":5,  "captain":"Hardik Pandya",  "coach":"Mahela Jayawardene", "home":"Wankhede Stadium",          "founded":2008},
    "Royal Challengers Bangalore": {"titles":1,  "captain":"Rajat Patidar",  "coach":"Andy Flower",        "home":"M Chinnaswamy Stadium",     "founded":2008},
    "Kolkata Knight Riders":       {"titles":3,  "captain":"Ajinkya Rahane", "coach":"Chandrakant Pandit", "home":"Eden Gardens",              "founded":2008},
    "Rajasthan Royals":            {"titles":1,  "captain":"Sanju Samson",   "coach":"Rahul Dravid",       "home":"Sawai Mansingh Stadium",    "founded":2008},
    "Sunrisers Hyderabad":         {"titles":1,  "captain":"Pat Cummins",    "coach":"Daniel Vettori",     "home":"Rajiv Gandhi Stadium",      "founded":2013},
    "Delhi Daredevils":            {"titles":0,  "captain":"Rishabh Pant",   "coach":"Hemang Badani",      "home":"Arun Jaitley Stadium",      "founded":2008},
    "Gujarat Titans":              {"titles":1,  "captain":"Shubman Gill",   "coach":"Ashish Nehra",       "home":"Narendra Modi Stadium",     "founded":2022},
    "Lucknow Super Giants":        {"titles":0,  "captain":"Rishabh Pant",   "coach":"Justin Langer",      "home":"Ekana Stadium",             "founded":2022},
    "Kings XI Punjab":             {"titles":0,  "captain":"Shreyas Iyer",   "coach":"Ricky Ponting",      "home":"Mohali Stadium",            "founded":2008},
}

IPL_WINNERS = {
    2008:"Rajasthan Royals",      2009:"Deccan Chargers",
    2010:"Chennai Super Kings",   2011:"Chennai Super Kings",
    2012:"Kolkata Knight Riders", 2013:"Mumbai Indians",
    2014:"Kolkata Knight Riders", 2015:"Mumbai Indians",
    2016:"Sunrisers Hyderabad",   2017:"Mumbai Indians",
    2018:"Chennai Super Kings",   2019:"Mumbai Indians",
    2020:"Mumbai Indians",        2021:"Chennai Super Kings",
    2022:"Gujarat Titans",        2023:"Chennai Super Kings",
    2024:"Kolkata Knight Riders", 2025:"Royal Challengers Bangalore",
}

# =========================================================
# HOME
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# SUPPRESS CHROME DEVTOOLS REQUEST
# =========================================================
@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def devtools():
    return "", 204

# =========================================================
# MATCH WINNER PREDICTION
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    data          = request.get_json()
    team1         = normalize(data["team1"])
    team2         = normalize(data["team2"])
    toss_winner   = normalize(data["toss_winner"])
    toss_decision = data["toss_decision"].lower()
    venue         = data["venue"]

    features = [
        encoders["team1"].transform([team1])[0],
        encoders["team2"].transform([team2])[0],
        encoders["toss_winner"].transform([toss_winner])[0],
        encoders["toss_decision"].transform([toss_decision])[0],
        encoders["venue"].transform([venue])[0],
    ]
    X        = np.array(features).reshape(1, -1)
    X        = scaler.transform(X)
    X        = X.reshape((1, 1, X.shape[1]))
    probs    = model.predict(X, verbose=0)[0]
    classes  = encoders["winner"].inverse_transform(np.arange(len(probs)))
    prob_map = dict(zip(classes, probs))
    allowed  = {
        team1: float(prob_map.get(team1, 0)),
        team2: float(prob_map.get(team2, 0)),
    }
    total  = allowed[team1] + allowed[team2]
    t1p    = allowed[team1] / total if total else 0.5
    t2p    = allowed[team2] / total if total else 0.5
    winner = team1 if t1p > t2p else team2
    return jsonify({
        "predicted_winner": winner,
        "confidence":   round(max(t1p, t2p), 4),
        "team1":        team1,
        "team2":        team2,
        "team1_prob":   round(t1p, 4),
        "team2_prob":   round(t2p, 4),
        "key_factors":  [f"{toss_winner} won toss", "Recent form considered", "Venue history used"],
    })

# =========================================================
# SCORE PREDICTION
# =========================================================
@app.route("/predict-score", methods=["POST"])
def predict_score():
    data    = request.get_json()
    bt      = normalize(data["batting_team"])
    bl      = normalize(data["bowling_team"])
    venue   = data["venue"]
    overs   = float(data["overs"])
    runs    = int(data["runs"])
    wickets = int(data["wickets"])
    last5   = int(data["last5"])

    bt_enc = score_team_encoder.transform([bt])[0]
    bl_enc = score_team_encoder.transform([bl])[0]
    v_enc  = score_venue_encoder.transform([venue])[0]

    X    = np.array([[bt_enc, bl_enc, v_enc, overs, runs, wickets, last5]])
    X    = score_scaler.transform(X)
    X    = X.reshape((1, 1, X.shape[1]))
    pred = int(score_model.predict(X, verbose=0)[0][0])
    return jsonify({
        "predicted_score": pred,
        "min_score":    pred - 15,
        "max_score":    pred + 15,
        "run_rate":     round(runs / overs, 2) if overs > 0 else 0,
        "wickets_left": 10 - wickets,
    })

# =========================================================
# HEAD TO HEAD
# =========================================================
@app.route("/head-to-head", methods=["POST"])
def head_to_head():
    data   = request.get_json()
    team_a = normalize(data["team_a"])
    team_b = normalize(data["team_b"])

    match_df = (
        DF[[
            "match_id", "batting_team", "bowling_team",
            "winner", "venue", "toss_winner", "toss_decision", "season"
        ]]
        .drop_duplicates(subset=["match_id"])
    )

    h2h = match_df[
        ((match_df["batting_team"] == team_a) | (match_df["bowling_team"] == team_a)) &
        ((match_df["batting_team"] == team_b) | (match_df["bowling_team"] == team_b))
    ].copy()

    total  = len(h2h)
    a_wins = int((h2h["winner"] == team_a).sum())
    b_wins = int((h2h["winner"] == team_b).sum())

    recent = []
    for _, row in h2h.tail(5).iterrows():
        recent.append({
            "season": str(row.get("season", "")),
            "winner": str(row.get("winner", "")),
        })

    by_venue = {}
    for venue, grp in h2h.groupby("venue"):
        by_venue[str(venue)] = {
            team_a: int((grp["winner"] == team_a).sum()),
            team_b: int((grp["winner"] == team_b).sum()),
        }

    return jsonify({
        "total":    total,
        "team_a":   team_a,
        "team_b":   team_b,
        "a_wins":   a_wins,
        "b_wins":   b_wins,
        "recent":   recent,
        "by_venue": by_venue,
    })

# =========================================================
# PLAYER VS PLAYER
# =========================================================
@app.route("/player-vs-player", methods=["POST"])
def player_vs_player():
    data = request.get_json()
    p1   = data["player1"].strip()
    p2   = data["player2"].strip()

    def get_stats(name):
        bat   = DF[DF["batter"] == name]
        bowl  = DF[DF["bowler"] == name]
        runs  = int(bat["runs_batter"].sum())
        balls = len(bat)
        sr    = round(runs / balls * 100, 2) if balls else 0
        fours = int((bat["runs_batter"] == 4).sum())
        sixes = int((bat["runs_batter"] == 6).sum())
        wkts  = int(bowl["player_out"].notna().sum())
        b_bld = len(bowl)
        ovs   = b_bld / 6 if b_bld else 0
        econ  = round(bowl["runs_batter"].sum() / ovs, 2) if ovs else 0
        avg   = round(runs / max(DF[DF["batter"] == name]["match_id"].nunique(), 1), 2)
        return {
            "name":    name,
            "runs":    runs,
            "sr":      sr,
            "fours":   fours,
            "sixes":   sixes,
            "wickets": wkts,
            "economy": econ,
            "average": avg,
            "matches": int(DF[(DF["batter"] == name) | (DF["bowler"] == name)]["match_id"].nunique()),
        }

    return jsonify({"player1": get_stats(p1), "player2": get_stats(p2)})

# =========================================================
# VENUE STATS
# =========================================================
@app.route("/venue-stats")
def venue_stats():
    match_df = (
        DF[["match_id", "venue", "batting_team", "bowling_team", "winner", "toss_decision"]]
        .drop_duplicates("match_id")
    )

    venues = []
    for venue, grp in match_df.groupby("venue"):
        total = len(grp)
        if total < 5:
            continue
        bat_first = grp[grp["toss_decision"] == "bat"]
        bat_wins  = int((bat_first["winner"] == bat_first["batting_team"]).sum()) if len(bat_first) else 0
        fld_first = grp[grp["toss_decision"] == "field"]
        fld_wins  = int((fld_first["winner"] == fld_first["bowling_team"]).sum()) if len(fld_first) else 0
        bat_pct   = round(bat_wins / len(bat_first) * 100, 1) if len(bat_first) else 0
        fld_pct   = round(fld_wins / len(fld_first) * 100, 1) if len(fld_first) else 0

        venue_balls = DF[DF["venue"] == venue]
        avg_score   = round(
            venue_balls.groupby("match_id")["runs_batter"].sum().mean(), 0
        ) if len(venue_balls) else 0

        venues.append({
            "venue":               venue,
            "total_matches":       total,
            "bat_first_win_pct":   bat_pct,
            "field_first_win_pct": fld_pct,
            "avg_score":           int(avg_score),
        })

    venues.sort(key=lambda x: x["total_matches"], reverse=True)
    return jsonify(venues[:12])

# =========================================================
# TOP PLAYERS (by team)
# =========================================================
@app.route("/players/top/<team_code>")
def top_players(team_code):
    team_code = team_code.upper()
    if team_code not in TEAM_MAP:
        return jsonify({"error": "invalid team"})
    team    = normalize(TEAM_MAP[team_code])
    team_df = DF[(DF["batting_team"] == team) | (DF["bowling_team"] == team)]
    
    bat = (
        team_df.groupby("batter")["runs_batter"]
        .sum()
        .where(lambda x: x > 0)        # ← filter zero runs
        .dropna()                        # ← drop zeros
        .sort_values(ascending=False)
        .head(10)
    )
    bowl = (
        team_df[team_df["player_out"].notna()]
        .groupby("bowler")["player_out"].count()
        .where(lambda x: x > 0)         # ← filter zero wickets
        .dropna()
        .sort_values(ascending=False)
        .head(10)
    )
    return jsonify({"top_batters": bat.to_dict(), "top_bowlers": bowl.to_dict()})


@app.route("/players/top-season/<team_code>/<season>")
def top_players_season(team_code, season):
    team_code = team_code.upper()
    if team_code not in TEAM_MAP:
        return jsonify({"error": "Invalid team"}), 400
    team = normalize(TEAM_MAP[team_code])
    df   = DF[DF["season"].astype(str) == str(season)].copy()
    if df.empty:
        return jsonify({"message": f"No data found for season {season}"}), 200
    
    bat = (
        df[df["batting_team"] == team]
        .groupby("batter")["runs_batter"]
        .sum()
        .where(lambda x: x > 0)         # ← filter zero runs
        .dropna()
        .sort_values(ascending=False)
        .head(15)
    )
    bowl = (
        df[(df["bowling_team"] == team) & (df["player_out"].notna())]
        .groupby("bowler")["player_out"].count()
        .where(lambda x: x > 0)         # ← filter zero wickets
        .dropna()
        .sort_values(ascending=False)
        .head(15)
    )
    return jsonify({"top_batters": bat.to_dict(), "top_bowlers": bowl.to_dict()})
# =========================================================
# AUTOSUGGEST — 2-stage optimized search
# Stage 1 : prefix + substring  (fast, no scoring)
# Stage 2 : fuzzy only if < 3 results
# =========================================================
@app.route("/api/suggest")
def suggest():
    q = request.args.get("q", "").strip().lower()
    if not q or len(q) < 2:
        return jsonify([])

    # Stage 1a — prefix
    results = [
        PLAYER_LOWER_INDEX[name]
        for name in PLAYER_LOWER_LIST
        if name.startswith(q)
    ]

    # Stage 1b — substring
    if len(results) < 5:
        for name in PLAYER_LOWER_LIST:
            orig = PLAYER_LOWER_INDEX[name]
            if q in name and orig not in results:
                results.append(orig)
            if len(results) >= 8:
                break

    # Deduplicate
    seen = set()
    final = []
    for r in results:
        if r not in seen:
            seen.add(r)
            final.append(r)
        if len(final) >= 8:
            break

    # Stage 2 — fuzzy fallback
    if len(final) < 3:
        fuzzy_matches = process.extract(q, ALL_PLAYERS, scorer=fuzz.WRatio, limit=8)
        for match, score, _ in fuzzy_matches:
            if score > 40 and match not in seen:
                final.append(match)
                seen.add(match)
            if len(final) >= 8:
                break

    return jsonify(final[:8])

# =========================================================
# PLAYER API
# =========================================================
@app.route("/api/player")
def player_api():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "no name"})
    bat     = DF[DF["batter"] == name]
    bowl    = DF[DF["bowler"] == name]
    runs    = int(bat["runs_batter"].sum())
    balls   = len(bat)
    sr      = round(runs / balls * 100, 2) if balls else 0
    fours   = int((bat["runs_batter"] == 4).sum())
    sixes   = int((bat["runs_batter"] == 6).sum())
    matches = int(DF[(DF["batter"] == name) | (DF["bowler"] == name)]["match_id"].nunique())
    wkts    = int(bowl["player_out"].notna().sum())
    ovs     = len(bowl) / 6 if len(bowl) else 0
    econ    = round(bowl["runs_batter"].sum() / ovs, 2) if ovs else 0
    return jsonify({
        "name":        name,
        "runs":        runs,
        "strike_rate": sr,
        "fours":       fours,
        "sixes":       sixes,
        "matches":     matches,
        "wickets":     wkts,
        "economy":     econ,
    })

# =========================================================
# ALL-TIME TOP PLAYERS
# =========================================================
@app.route("/players/top-all")
def top_all_players():
    bat  = (
        DF.groupby("batter")["runs_batter"]
        .sum().sort_values(ascending=False).head(10)
    )
    bowl = (
        DF[DF["player_out"].notna()]
        .groupby("bowler")["player_out"].count()
        .sort_values(ascending=False).head(10)
    )
    return jsonify({
        "batters": [{"name": n, "runs":    int(r)} for n, r in bat.items()],
        "bowlers": [{"name": n, "wickets": int(w)} for n, w in bowl.items()],
    })

# =========================================================
# TEAM INFO
# =========================================================
@app.route("/team-info/<team_code>")
def team_info(team_code):
    team_code = team_code.upper()
    if team_code not in TEAM_MAP:
        return jsonify({"error": "Invalid team"})
    team = normalize(TEAM_MAP[team_code])
    return jsonify({"team": team, "info": TEAM_INFO.get(team, {})})

# =========================================================
# TOSS HISTORY
# =========================================================
@app.route("/toss-history", methods=["POST"])
def toss_history():
    data          = request.get_json()
    team_a        = normalize(data["team_a"])
    team_b        = normalize(data["team_b"])
    venue         = data["venue"]
    match_level   = DF[["match_id", "venue", "toss_winner"]].drop_duplicates("match_id")
    venue_matches = match_level[match_level["venue"] == venue]
    toss_wins     = venue_matches["toss_winner"].value_counts().to_dict()
    return jsonify({
        "toss_wins": {
            team_a: toss_wins.get(team_a, 0),
            team_b: toss_wins.get(team_b, 0),
        }
    })

# =========================================================
# SEASON WINNERS
# =========================================================
@app.route("/season-winners")
def season_winners():
    return jsonify(IPL_WINNERS)


# =========================================================
# DREAM TEAM BUILDER
# Scoring algorithm:
#   Batters  : runs + (SR/100)*200 + sixes*4 + fours*2
#   Bowlers  : wickets*30 + (8-economy)*15   (economy bonus)
#   All-round: bat_score + bowl_score * 0.7
#   WK bonus : +50 if player has kept (approximated by bat position)
# =========================================================

@app.route("/dream-team/test", methods=["GET"])
def dream_team_test():
    return jsonify({"status": "ok", "message": "POST to /dream-team with team_a, team_b, format"})

@app.route("/dream-team", methods=["POST"])
def dream_team():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        team_a = normalize(data["team_a"])
        team_b = normalize(data["team_b"])
        fmt    = data.get("format", "overall")

        df = DF.copy()
        if fmt == "recent":
            seasons = sorted(df["season"].dropna().unique())[-3:]
            df = df[df["season"].isin(seasons)]
        elif fmt not in ("overall", "recent"):
            df = df[df["season"].astype(str) == str(fmt)]

        def get_player_stats(name, team):
            bat   = df[df["batter"] == name]
            bowl  = df[df["bowler"] == name]
            runs  = int(bat["runs_batter"].sum())
            balls = len(bat)
            sr    = round(runs / balls * 100, 2) if balls else 0
            fours = int((bat["runs_batter"] == 4).sum())
            sixes = int((bat["runs_batter"] == 6).sum())
            matches = int(df[(df["batter"] == name) | (df["bowler"] == name)]["match_id"].nunique())
            wkts  = int(bowl["player_out"].notna().sum())
            b_bld = len(bowl)
            ovs   = b_bld / 6 if b_bld else 0
            econ  = round(bowl["runs_batter"].sum() / ovs, 2) if ovs else 0

            KNOWN_WK = {
                "MS Dhoni", "Rishabh Pant", "KL Rahul", "Sanju Samson",
                "Dinesh Karthik", "Wriddhiman Saha", "Robin Uthappa",
                "Parthiv Patel", "Quinton de Kock", "Jos Buttler",
                "Heinrich Klaasen", "Ishan Kishan", "Naman Ojha",
                "Adam Gilchrist", "Kumar Sangakkara", "Brad Haddin",
                "Matthew Wade", "AB de Villiers", "Phil Salt",
                "Nicholas Pooran", "Jonny Bairstow", "Devon Conway",
                "Rahul Tripathi", "Sheldon Jackson", "Tim Seifert",
                "Sam Billings", "Liam Livingstone",
            }
            is_wk = name in KNOWN_WK
            if not is_wk and "wicket_kind" in df.columns and "fielders" in df.columns:
                stump_rows = df[df["wicket_kind"] == "stumped"]["fielders"].dropna()
                stumped_by = set()
                for val in stump_rows:
                    stumped_by.add(str(val).strip())
                if name in stumped_by:
                    is_wk = True

            bat_score   = runs + (sr / 100) * 200 + sixes * 4 + fours * 2
            bowl_score  = wkts * 30 + max(0, (8 - econ)) * 15 if ovs > 0 else 0
            total_score = bat_score + bowl_score * 0.7 + (50 if is_wk else 0)

            return {
                "name": name, "team": team, "runs": runs, "sr": sr,
                "fours": fours, "sixes": sixes, "wickets": wkts,
                "economy": econ, "matches": matches, "is_wk": is_wk,
                "bat_score":     round(bat_score, 1),
                "bowl_score":    round(bowl_score, 1),
                "fantasy_score": round(total_score, 1),
                "selected": False,
            }

        def top_squad(team):
            bat_df  = df[df["batting_team"] == team]
            bowl_df = df[df["bowling_team"] == team]

            bat_players  = (bat_df.groupby("batter")["match_id"]
                            .nunique().sort_values(ascending=False).head(15).index.tolist())
            bowl_players = (bowl_df[bowl_df["player_out"].notna()]
                            .groupby("bowler")["match_id"]
                            .nunique().sort_values(ascending=False).head(10).index.tolist())

            all_names = list(dict.fromkeys(bat_players + bowl_players))
            return [get_player_stats(p, team) for p in all_names]

        def classify(p):
            if p["is_wk"]: return "WK"
            r, w = p["runs"], p["wickets"]
            has_runs = r > 200
            has_wkts = w > 5
            if has_runs and has_wkts: return "ALL"
            if has_wkts and not has_runs: return "BOWL"
            return "BAT"

        squad_a     = top_squad(team_a)
        squad_b     = top_squad(team_b)
        all_players = squad_a + squad_b
        all_players.sort(key=lambda x: x["fantasy_score"], reverse=True)

        xi          = []
        counts      = {"WK": 0, "BAT": 0, "ALL": 0, "BOWL": 0}
        team_counts = {team_a: 0, team_b: 0}
        comp   = data.get("composition", {})
        limits = {
            "WK":   int(comp.get("wk",   1)),
            "BAT":  int(comp.get("bat",  4)),
            "ALL":  int(comp.get("all",  2)),
            "BOWL": int(comp.get("bowl", 4)),
        }

        wk_candidates = sorted(
            [p for p in all_players if p["is_wk"]],
            key=lambda x: x["fantasy_score"], reverse=True
        )
        if not wk_candidates:
            KNOWN_WK = {
                "MS Dhoni","Rishabh Pant","KL Rahul","Sanju Samson",
                "Dinesh Karthik","Wriddhiman Saha","Robin Uthappa",
                "Parthiv Patel","Quinton de Kock","Jos Buttler",
                "Heinrich Klaasen","Ishan Kishan","Naman Ojha",
                "Adam Gilchrist","Kumar Sangakkara","Brad Haddin",
                "Matthew Wade","Phil Salt","Nicholas Pooran",
                "Jonny Bairstow","Devon Conway","Tim Seifert",
                "Sam Billings","Sheldon Jackson",
            }
            for p in all_players:
                if p["name"] in KNOWN_WK:
                    p["is_wk"] = True
                    wk_candidates.append(p)

        if wk_candidates:
            best_wk = wk_candidates[0]
            best_wk["selected"] = True
            xi.append(best_wk)
            counts["WK"] = 1
            team_counts[best_wk["team"]] = team_counts.get(best_wk["team"], 0) + 1

        for p in all_players:
            if len(xi) >= 11: break
            if any(x["name"] == p["name"] for x in xi): continue
            role = classify(p)
            tc   = team_counts.get(p["team"], 0)
            if counts[role] < limits[role] and tc < 7:
                counts[role] += 1
                team_counts[p["team"]] = tc + 1
                p["selected"] = True
                xi.append(p)

        selected_names = {p["name"] for p in xi}
        for p in all_players:
            if len(xi) >= 11: break
            if p["name"] not in selected_names:
                tc = team_counts.get(p["team"], 0)
                if tc < 7:
                    p["selected"] = True
                    team_counts[p["team"]] = tc + 1
                    xi.append(p)

        xi.sort(key=lambda x: x["fantasy_score"], reverse=True)
        captain      = xi[0]["name"] if xi else ""
        vice_captain = xi[1]["name"] if len(xi) > 1 else ""

        xi_names = {p["name"] for p in xi}
        for p in squad_a + squad_b:
            p["selected"] = p["name"] in xi_names

        return jsonify({
            "xi":             xi,
            "captain":        captain,
            "vice_captain":   vice_captain,
            "team_a_players": sorted(squad_a, key=lambda x: x["fantasy_score"], reverse=True),
            "team_b_players": sorted(squad_b, key=lambda x: x["fantasy_score"], reverse=True),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)