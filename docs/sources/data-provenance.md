# Data sources and provenance

## 1. Zenodo — COICOP 2018 classification, manual records

- **Record**: https://zenodo.org/records/18459651 (DOI 10.5281/zenodo.18459651)
- **Title**: "COICOP 2018 classification - manual and synthetic"
- **Licence**: CC-BY-SA 4.0
- **Files**: `manual_labels_coicop2018.csv` (4,993 rows: `name`, `category`, `code`) —
  **only this manual file is used**; the synthetic file in the same record is *not* used.
- **Retrieved**: 2026-09-01, via the record's `files-archive` API endpoint
  (`https://zenodo.org/api/records/18459651/files-archive`; the direct browser link
  returns 403 to scripted clients).
- **Pitfalls noted**:
  - the `category` column mixes Italian retailer-style hierarchies and English
    web-scrape-style breadcrumbs — realistic, kept as-is;
  - product names are multilingual (mostly Italian, some English/German);
  - `code` is COICOP 2018 at the subclass (5-digit) level.
- **Verification**: md5 of the file inside the archive matches the checksum listed on
  the record page at retrieval time.

## 2. UNSD — COICOP 2018 structure

- **URL**: https://unstats.un.org/unsd/classifications/Econ/Download/COICOP_2018_English_structure.xlsx
- **Content**: 871 rows × 6 columns (`code`, `title`, `intro`, `includes`, `alsoIncludes`,
  `excludes`) — the full official English structure of COICOP 2018.
- **Retrieved**: 2026-09-01 (HTTP 200, 111,269 bytes).
- **Usage in this repo**: `code`/`title` for class labels everywhere; titles are also used
  as zero-shot NLI hypotheses in the notebook (§7).
- **Notes**: the workbook has a single sheet `COICOP_2018`; titles of subclasses carry a
  trailing "(ND)" durability marker that `src/prepare_data.py` strips for display.

## 3. UN-CEBD Scanner Data Wiki (methodology, not data)

- **Space**: `GWGSD` ("UN-CEBD - Scanner Data Wiki") on https://unstats.un.org/wiki/
- **Chapter**: *Handbook on utilising new data sources in the production of consumer
  price statistics* → Classification (page id 240910479, version v7, 2025-04-28) and its
  11 sub-pages (Methods 0-4, evaluation, imbalance, operational best practices).
- **Retrieved**: 2026-09-01 via the Confluence REST API
  (`/rest/api/space/GWGSD/content/page`, `/rest/api/content/{id}?expand=body.storage`).
- The chapter states that official companion notebooks are published in
  https://github.com/UN-Task-Team-for-Scanner-Data/classification_methods (checked
  2026-09-01: repository exists; `notebooks/` still contains only `.gitkeep`).

## 4. Synthetic / augmented components (created by this repo)

`src/prepare_data.py` deterministically (numpy `default_rng(42)`) generates:

- `banner` — outlet chain sampled from 4 fictional banners with class-tilted probabilities;
- `is_promo` — promotion flag with class-specific rates (0.12–0.35);
- `gtin` — synthetic 13-digit codes (2-digit class prefix + stable hash of the name);
- `first_seen_month` — per-class 70/30 January/February split (shuffled, seeded);
- 25 hand-written new products (March churn batch) including one out-of-scope energy
  drink, listed in `src/prepare_data.py` with their held-aside truth labels.

All synthetic values are **fictional** and exist only to demonstrate attribute-based and
production-loop techniques; they carry no statistical claim about real markets.
