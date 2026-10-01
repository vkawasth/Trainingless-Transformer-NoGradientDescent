"""AllSides (Baly et al. 2020, Article-Bias-Prediction) loader: Trump-directed stance without training.

Data: git clone --depth 1 https://github.com/ramybaly/Article-Bias-Prediction ; set ALLSIDES_DIR to its data/jsons
(default data/jsons here). Not redistributed.
Per article, over its sentences that mention Trump (fixed lexicons; no model is trained):
  adv    ADVERSARIAL framing rate: share of Trump sentences with fact-check / condemnation terms (falsely, without
         evidence, baseless, lie(d), misleading, unfounded, racist, authoritarian, ...). Target-directed.
  hon    HONORIFIC share: "President (Donald) Trump" / all "Trump" mentions (meaningful after 2017-01-20).
  tone   mean VADER compound (general TONE of Trump sentences; NOT favourability: right outlets score lower,
         because their Trump sentences quote attacks on opponents).
  dir    DIRECTED fact-check sentences: "Trump ... falsely / lied / lies / baseless / without evidence / misleading /
         unfounded / untrue" with Trump within 3 words before the term (high precision, Trump-directed).
Outcome of an article = adversarial (any adversarial Trump sentence) 1 / 0; outcome_dir = any directed sentence.
Caveat: the broad adversarial vocabulary is not always aimed at Trump (right outlets after the Mueller report use it
for the investigation: "baseless collusion claims"); the directed marker is the target-directed one.
Sources = AllSides side (left / center / right) or outlet. Topic cells = coarse groups of AllSides topics.
Event unit = (date, AllSides topic) group, AllSides' same-story grouping.
"""
import os, json, glob, re, datetime, pickle
import numpy as np

DIR = os.environ.get("ALLSIDES_DIR", os.path.join(os.path.dirname(__file__), "data", "jsons"))
CACHE = os.path.join(os.path.dirname(__file__), "allsides_cache.pkl")
TOPIC_GROUP = {
    "elections": "elections", "politics": "politics", "white_house": "politics", "republican_party": "politics",
    "democratic_party": "politics", "polarization": "politics", "media_bias": "media",
    "immigration": "immigration", "healthcare": "healthcare", "coronavirus": "coronavirus",
    "middle_east": "foreign", "foreign_policy": "foreign", "world": "foreign", "trade": "foreign",
    "national_security": "foreign", "terrorism": "foreign", "north_korea": "foreign", "russia": "foreign", "china": "foreign",
    "fbi": "justice", "impeachment": "justice", "supreme_court": "justice", "justice": "justice", "justice_department": "justice",
    "economy_and_jobs": "economy", "taxes": "economy", "federal_budget": "economy", "business": "economy",
}
OUTLET = {"Fox Online News": "Fox", "Fox News": "Fox", "CNN (Web News)": "CNN", "New York Times - News": "NYT",
          "Washington Times": "WashTimes", "NPR Online News": "NPR", "Politico": "Politico", "Breitbart News": "Breitbart",
          "Vox": "Vox", "The Hill": "TheHill", "USA TODAY": "USAToday", "Reuters": "Reuters", "Townhall": "Townhall",
          "National Review": "NatReview", "Salon": "Salon", "Newsmax": "Newsmax"}
_split = re.compile(r"(?<=[.!?])\s+")
ADV = re.compile(r"\b(falsely|without evidence|baseless(ly)?|false claims?|lie[sd]?|lying|misleading|unfounded|debunked|"
                 r"racist|xenophob\w*|authoritarian)\b", re.I)
DIRECTED = re.compile(r"\bTrump(?:'s|’s)?\s+(?:\w+\s+){0,3}?(falsely|lied|lies|lying|false(?:ly)? claim\w*|baseless(?:ly)?|"
                      r"without (?:providing )?evidence|misleading|unfounded|untrue|inaccurate(?:ly)?)\b", re.I)


def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer(); rows = []
    for f in sorted(glob.glob(f"{DIR}/*.json")):
        d = json.load(open(f)); dt = d.get("date") or ""
        if not re.match(r"^20(1[2-9]|20)-\d\d-\d\d$", dt):
            continue
        sents = [s for s in _split.split(d["content_original"]) if "trump" in s.lower()]
        if not sents:
            continue
        sc = float(np.mean([an.polarity_scores(s)["compound"] for s in sents]))
        nadv = sum(bool(ADV.search(s)) for s in sents); ndir = sum(bool(DIRECTED.search(s)) for s in sents)
        txt = " ".join(sents)
        ntr = len(re.findall(r"\bTrump\b", txt)); npres = len(re.findall(r"\bPresident (Donald )?Trump\b", txt))
        rows.append(dict(date=datetime.date.fromisoformat(dt), topic=d["topic"], group=TOPIC_GROUP.get(d["topic"], "other"),
                         side=d["bias_text"], outlet=OUTLET.get(d["source"], d["source"]), tone=sc, n_sent=len(sents),
                         n_adv=nadv, adv=nadv / len(sents), outcome=int(nadv > 0), n_dir=ndir, outcome_dir=int(ndir > 0), n_trump=ntr, n_pres=npres,
                         hon=npres / ntr if ntr else np.nan))
    pickle.dump(rows, open(CACHE, "wb"))
    return rows


if __name__ == "__main__":
    R = load(); import collections
    print(len(R), "Trump-mentioning dated articles")
    print(collections.Counter(r["side"] for r in R)); print(collections.Counter(r["group"] for r in R))
    for s in ("left", "center", "right"):
        x = [r for r in R if r["side"] == s]
        print(s, len(x), "tone", round(np.mean([r["tone"] for r in x]), 3), "adv", round(np.mean([r["adv"] for r in x]), 4),
              "P(adv article)", round(np.mean([r["outcome"] for r in x]), 3),
              "P(directed article)", round(np.mean([r["outcome_dir"] for r in x]), 4))
