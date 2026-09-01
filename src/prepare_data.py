"""Prepare the demo dataset for the COICOP classification notebook.

Pipeline (run once, outputs committed to the repo):

1. Filter the Zenodo *manual* labels to a 15-class subset spanning food divisions
   01 and 02 (imbalanced on purpose: 584 -> 42 rows per class).
2. Extract structured attributes from the product names (brand, pack size,
   organic flag) and deterministically augment the attributes that scanner
   feeds typically carry but the Zenodo file lacks: outlet-chain banner,
   promotion flag, GTIN, and first-seen month (probabilities per COICOP class,
   fixed RNG seed).
3. Build a Method-1 mapping table (retailer category -> COICOP) from the
   training month, with purity statistics.
4. Sample a synthetic production batch of new unique products (churn) from the
   held-out month, same augmented schema, plus a few hand-written new products.

Deterministic: pandas + numpy default_rng(seed=42). No network access required
once data/raw/manual_labels_coicop2018.csv is in place.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RAW = "data/raw/manual_labels_coicop2018.csv"
COICOP_XLSX = "data/raw/COICOP_2018_English_structure.xlsx"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Class subset: 15 COICOP 2018 subclasses covering divisions 01 + 02
# ---------------------------------------------------------------------------
TARGET_CODES = [
    "01.1.2.2",  # Meat, fresh, chilled or frozen
    "01.1.2.3",  # Meat, dried, salted, in brine or smoked
    "01.1.2.5",  # Preparations of meat, offal or blood
    "01.1.6.2",  # Citrus fruits, fresh
    "01.1.6.3",  # Stone fruits and pome fruits, fresh
    "01.1.6.4",  # Berries, fresh
    "01.1.6.5",  # Other fruits, fresh
    "01.1.6.9",  # Fruits and nuts, ground, and in other preparations
    "01.1.7.1",  # Leafy or stem vegetables, fresh or chilled
    "01.1.7.2",  # Fruit-bearing vegetables, fresh or chilled
    "01.1.7.6",  # Pulses
    "01.1.9.1",  # Ready-made food
    "01.2.1.0",  # Fruit and vegetable juices
    "01.2.3.0",  # Tea, maté and infusions
    "02.1.2.1",  # Wine from grapes
]

df = pd.read_csv(RAW, dtype={"code": str})
sub = df[df["code"].isin(TARGET_CODES)].copy()
print(f"Subset: {len(sub)} rows, {sub['code'].nunique()} classes (target {len(TARGET_CODES)})")
assert sub["code"].nunique() == len(TARGET_CODES)

# COICOP titles from the official UNSD structure file
co = pd.read_excel(COICOP_XLSX)
co["code"] = co["code"].astype(str)
titles = dict(zip(co["code"], co["title"].astype(str).str.replace(r"\s*\(ND\)$", "", regex=True)))
sub["coicop_title"] = sub["code"].map(titles)

# ---------------------------------------------------------------------------
# 2. Attribute extraction + deterministic augmentation
# ---------------------------------------------------------------------------
# 2a. Brand: first token when it matches a curated brand list, else null
KNOWN_BRANDS = {
    "conad", "carrefour", "bonduelle", "citterio", "simmenthal", "aia", "amadori",
    "primia", "f&v", "rio", "dimmidisi", "noberasco", "innocent", "rauch", "pfanner",
    "colfiorito", "fratelli", "casa", "san", "la", "el", "il", "valfrutta", "cottin",
    "percorso", "sapori", "grevensteiner", "hell", "elledi", "hag", "nescafe",
    "rocca", "distilleria", "marzadro", "ferrero", "barilla", "rider",
}
BRAND_LOOKUP = {b: b.capitalize() for b in KNOWN_BRANDS}


def extract_brand(name: str) -> str | None:
    toks = str(name).lower().split()
    if toks and toks[0] in KNOWN_BRANDS:
        return BRAND_LOOKUP[toks[0]]
    return None


sub["brand"] = sub["name"].apply(extract_brand)

# 2b. Pack size parsed from the product name
SIZE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|gr?|grams?|ml|lt?|cl)\b", re.I)
UNITS = {"kg": "kg", "g": "g", "gr": "g", "gram": "g", "grams": "g",
         "ml": "ml", "l": "l", "lt": "l", "cl": "cl"}


def extract_size(name: str) -> str | None:
    m = SIZE_RE.search(str(name))
    if not m:
        return None
    qty = m.group(1).replace(",", ".")
    unit = UNITS.get(m.group(2).lower())
    if unit is None:
        return None
    q = float(qty)
    # plausibility bounds per unit (retail grocery range)
    bounds = {"kg": (0.01, 50), "g": (1, 5000), "ml": (5, 5000),
              "l": (0.05, 10), "cl": (1, 100)}
    lo, hi = bounds[unit]
    if not (lo <= q <= hi):
        return None
    return f"{q:g} {unit}"


sub["pack_size"] = sub["name"].apply(extract_size)

# 2c. Organic flag from name keywords (Italian + English, as found in the data)
ORG_RE = re.compile(r"\b(bio|biologica|biologico|biologiche|biologici|organic)\b", re.I)
sub["is_organic"] = sub["name"].str.contains(ORG_RE, regex=True)
sub.loc[sub["category"].str.contains("biologiche", na=False), "is_organic"] = True

# 2d. Deterministic synthetic attributes absent from the Zenodo file:
#     GTIN, outlet banner, promotion flag, first-seen month
rng = np.random.default_rng(SEED)
BANNERS = ["SuperBuy", "HyperMarket", "DiscountOra", "CoopNord"]

sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle once
banner_assign, promo_assign, month_assign, gtin_assign = [], [], [], []
class_pos = {c: 0 for c in TARGET_CODES}
for i, row in sub.iterrows():
    code = row["code"]
    idx = TARGET_CODES.index(code)
    # banner: probability vector rotated by class index (regional assortment tilt)
    base = np.array([0.42, 0.26, 0.18, 0.14])
    tilt = np.roll(base, idx % 4) * (1 + 0.10 * (idx % 3))
    p = tilt / tilt.sum()
    banner_assign.append(rng.choice(BANNERS, p=p))
    # promo: class-specific promotion intensity
    promo_p = {"01.1.9.1": 0.35, "01.2.1.0": 0.30, "02.1.2.1": 0.28, "01.1.2.5": 0.25,
               "01.2.3.0": 0.18, "01.1.7.2": 0.18, "01.1.6.9": 0.15, "01.1.6.3": 0.15}.get(code, 0.12)
    promo_assign.append(bool(rng.random() < promo_p))
    # first-seen month: first 70% of each class (in shuffled order) = Jan, rest = Feb
    pos = class_pos[code]
    n_class = (sub["code"] == code).sum()
    month_assign.append("2025-01" if pos < int(n_class * 0.7) else "2025-02")
    class_pos[code] = pos + 1
    # synthetic GTIN: 2-digit class prefix + 11 digits from a stable hash
    stable = int.from_bytes(row["name"].encode("utf-8"), "little") % (10**11)
    gtin_assign.append(f"{idx + 1:02d}{stable:011d}")

sub["banner"] = banner_assign
sub["is_promo"] = promo_assign
sub["first_seen_month"] = month_assign
sub["gtin"] = gtin_assign

cols = ["gtin", "name", "brand", "pack_size", "is_organic", "banner", "is_promo",
        "retailer_category", "coicop_code", "coicop_title", "first_seen_month"]
clean = sub.rename(columns={"category": "retailer_category", "code": "coicop_code"})[cols]
clean["brand"] = clean["brand"].fillna("")
clean["is_organic"] = clean["is_organic"].astype(bool)
clean["is_promo"] = clean["is_promo"].astype(bool)
clean.to_csv("data/processed/products_train.csv", index=False)
print(f"products_train.csv: {len(clean)} rows")
print(clean.head(3).to_string())

# ---------------------------------------------------------------------------
# 3. Method-1 mapping table from the training month only
# ---------------------------------------------------------------------------
train = clean[clean["first_seen_month"] == "2025-01"]
rows = []
for cat, g in train.groupby("retailer_category"):
    vc = g["coicop_code"].value_counts()
    rows.append({"retailer_category": cat, "n_products": len(g),
                 "mapped_code": vc.index[0], "purity": round(vc.iloc[0] / len(g), 3),
                 "n_codes": int(vc.size)})
mapping = pd.DataFrame(rows).sort_values("n_products", ascending=False).reset_index(drop=True)
mapping["mapped_title"] = mapping["mapped_code"].map(titles)
mapping.to_csv("data/processed/category_mapping_table.csv", index=False)
print(f"\ncategory_mapping_table.csv: {len(mapping)} categories")
print(mapping.head(8).to_string())

# ---------------------------------------------------------------------------
# 4. Synthetic production batch (churn): holdout-month rows + new products
# ---------------------------------------------------------------------------
prod = clean[clean["first_seen_month"] == "2025-02"].drop(
    columns=["coicop_code", "coicop_title"]).copy()
print(f"\nholdout month rows: {len(prod)}")

# Hand-written new products hitting known gaps in the rules:
#  - English-language names (new importer assortment)
#  - an out-of-scope energy drink (Classification -> filtering consideration)
#  - a promo-tagged name variant that breaks a keyword rule
NEW_PRODUCTS = [
    # name, retailer_category, banner, brand, pack, organic, promo, true COICOP (kept aside)
    ("Fresh Ribeye Steak Angus 300g", "carne bovina", "SuperBuy", "", "300 g", False, False, "01.1.2.2"),
    ("Organic Free-Range Chicken Fillets 500g", "carne avicunicola", "CoopNord", "", "500 g", True, False, "01.1.2.2"),
    ("Smoked Streaky Bacon 200g", "carne suina", "HyperMarket", "", "200 g", False, False, "01.1.2.3"),
    ("Salami Felino PGI 250g", "formaggi e salumi salumi confezionati", "SuperBuy", "", "250 g", False, False, "01.1.2.3"),
    ("Vienna Sausages in Brine 400g", "scatolame carne in gelatina", "DiscountOra", "", "400 g", False, True, "01.1.2.5"),
    ("Blood Pudding Traditional 180g", "carne suina", "CoopNord", "", "180 g", False, False, "01.1.2.5"),
    ("Sicilian Oranges 1kg Bag", "ortofrutta frutta fresca", "SuperBuy", "", "1 kg", False, False, "01.1.6.2"),
    ("Lemon Juice Not From Concentrate 250ml", "juices", "HyperMarket", "", "250 ml", False, False, "01.2.1.0"),
    ("Pink Grapefruit Pack of 4", "ortofrutta frutta fresca", "DiscountOra", "", "", False, True, "01.1.6.2"),
    ("Golden Delicious Apples 1kg", "ortofrutta frutta fresca", "CoopNord", "", "1 kg", False, False, "01.1.6.3"),
    ("Apricots Pescatore 500g", "ortofrutta frutta fresca", "SuperBuy", "", "500 g", False, False, "01.1.6.3"),
    ("Wild Blueberries 250g", "ortofrutta frutta fresca", "HyperMarket", "", "250 g", False, True, "01.1.6.4"),
    ("Pomegranate Each", "ortofrutta frutta fresca", "DiscountOra", "", "", False, False, "01.1.6.5"),
    ("Cherry Tomato Passata 700ml", "ortofrutta verdure", "CoopNord", "", "700 ml", False, False, "01.1.7.2"),
    ("Organic Rainbow Chard 200g", "ortofrutta verdure", "SuperBuy", "", "200 g", True, False, "01.1.7.1"),
    ("Green Lentils du Puy 500g", "ortofrutta legumi e cereali", "HyperMarket", "", "500 g", False, False, "01.1.7.6"),
    ("Borlotti Beans Canned 400g", "tins cans & packets", "DiscountOra", "", "400 g", False, False, "01.1.7.6"),
    ("Truffle Tagliatelle Ready Meal 350g", "tins cans & packets", "CoopNord", "", "350 g", False, True, "01.1.9.1"),
    ("Grilled Zucchini Ready Salad 250g", "ortofrutta verdure", "SuperBuy", "", "250 g", False, False, "01.1.9.1"),
    ("Earl Grey Tea 50 Teabags", "zucchero, caffe', te', infusi | te'", "HyperMarket", "", "", False, False, "01.2.3.0"),
    ("Chamomile and Lemon Infusion 20 Bags", "zucchero, caffe', te', infusi | te'", "DiscountOra", "", "", False, False, "01.2.3.0"),
    ("Primitivo di Manduria DOC 75cl", "wine", "CoopNord", "", "75 cl", False, True, "02.1.2.1"),
    ("Vermentino di Sardegna 2023 750ml", "vini, birra e liquori | vini", "SuperBuy", "", "750 ml", False, False, "02.1.2.1"),
    ("Rainforest Smoothie Mango 330ml", "juices & smoothies", "HyperMarket", "", "330 ml", False, False, "01.2.1.0"),
    ("Taurus Boost Energy Drink 500ml", "soft drinks", "DiscountOra", "", "500 ml", False, True, "OUT-OF-SCOPE"),
]

new_rng = np.random.default_rng(SEED + 1)
new_rows = []
for k, (name, cat, banner, brand, pack, org, promo, _truth) in enumerate(NEW_PRODUCTS):
    new_rows.append({
        "gtin": f"99{k + 1:011d}",
        "name": name,
        "brand": brand,
        "pack_size": pack if pack else None,
        "is_organic": org,
        "banner": banner,
        "is_promo": promo,
        "retailer_category": cat,
        "first_seen_month": "2025-03",
    })
new_df = pd.DataFrame(new_rows)
prod = pd.concat([prod, new_df], ignore_index=True)
prod.to_csv("data/processed/production_batch.csv", index=False)
print(f"production_batch.csv: {len(prod)} rows ({len(new_df)} hand-written new products)")

# ground truth for the new products only, used for the final validation demo
pd.DataFrame(NEW_PRODUCTS, columns=["name", "retailer_category", "banner", "brand",
                                    "pack_size", "is_organic", "is_promo", "true_coicop"]
             ).to_csv("data/processed/new_products_truth.csv", index=False)
print("new_products_truth.csv written")
