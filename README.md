# COICOP Classification Notebook

Example code for classifying scanner-data / web-scraped products into **COICOP 2018**
for **Consumer Price Index (CPI)** production, following the methodology of the
**UN-CEBD Task Team on Scanner Data**:

> *Handbook on utilising new data sources in the production of consumer price statistics* —
> [Classification chapter (UN Statistics Wiki, space `GWGSD`)](https://unstats.un.org/wiki/display/GWGSD/Classification)

The single notebook [`notebooks/coicop_classification_examples.ipynb`](notebooks/coicop_classification_examples.ipynb)
walks through **all five classification methods** of the handbook on one realistic dataset,
each evaluated on the same held-out month so results are directly comparable:

| # | Method (handbook page) | Demonstrated with |
|---|---|---|
| 0 | [Manual labelling / validation](https://unstats.un.org/wiki/display/GWGSD/Method+0%3A+Manual+labelling+or+validation+of+predicted+labels) | inter-rater agreement sample, bootstrap CI on validation accuracy |
| 1 | [Attribute-based (rule-based)](https://unstats.un.org/wiki/display/GWGSD/Method+1%3A+Attribute-based+classification+method) | retailer-category → COICOP mapping table as version-controlled data, purity filters, break reports |
| 2 | [Pattern matching](https://unstats.un.org/wiki/display/GWGSD/Method+2%3A+Pattern-matching+classification+method) | multilingual keyword decision-lists, keyword-boolean attributes, attribute imputation |
| 3 | [Recommendation / machine-assisted](https://unstats.un.org/wiki/display/GWGSD/Method+3%3A+Recommendation+%2F+Machine-assisted+classification) | LogisticRegression top-3 shortlist with probabilities (top-k coverage metrics) |
| 4 | [Machine learning](https://unstats.un.org/wiki/display/GWGSD/Method+4%3A+Machine+Learning+classification+method) | TF-IDF (word + char n-grams) + LogisticRegression / LinearSVC / ComplementNB, and **zero-shot NLI** with `facebook/bart-large-mnli` using official COICOP titles as hypotheses |

Cross-cutting handbook topics:

- [How to evaluate classification methods](https://unstats.un.org/wiki/display/GWGSD/How+to+evaluate+classification+methods):
  precision-first metrics (macro/micro, F-beta), confusion-matrix review — §8
- [Working with class imbalance](https://unstats.un.org/wiki/display/GWGSD/Working+with+class+imbalance):
  KS distribution test, class weighting, precision-first thresholds (selective editing) — §9
- [Operational best practices](https://unstats.un.org/wiki/display/GWGSD/Operational+best+practices):
  a mini **production loop** on a monthly churn batch — coverage report, break/drift detection,
  validation sample, retraining — §10

## Data

| File | Content | Origin |
|---|---|---|
| `data/raw/manual_labels_coicop2018.csv` | 4,993 real product names manually labelled to COICOP 2018 | Zenodo [10.5281/zenodo.18459651](https://zenodo.org/records/18459651) — **manual** records only, CC-BY-SA 4.0 |
| `data/raw/COICOP_2018_English_structure.xlsx` | official COICOP 2018 titles & scope notes | [UNSD download](https://unstats.un.org/unsd/classifications/Econ/Download/COICOP_2018_English_structure.xlsx) |
| `data/processed/products_train.csv` | 3,442 products, 15 COICOP classes, augmented attributes | built by `src/prepare_data.py` |
| `data/processed/category_mapping_table.csv` | Method-1 rules + purity stats (from January) | built by `src/prepare_data.py` |
| `data/processed/production_batch.csv` | February carry-over + 25 synthetic new products (churn) | built by `src/prepare_data.py` |
| `data/processed/new_products_truth.csv` | held-aside truth for the 25 new products (validation demo) | built by `src/prepare_data.py` |

The Zenodo file provides `name`, `category` (retailer-style), `code` (COICOP).
`src/prepare_data.py` adds the structured attributes a real scanner feed carries but the
Zenodo file lacks — **deterministically, seed=42**:

- parsed from names: `brand`, `pack_size`, `is_organic`
- augmented: `banner` (outlet chain), `is_promo`, `gtin`, `first_seen_month`
  (class-dependent probabilities, fixed RNG)

Classes span 584 → 45 products: deliberate, realistic **class imbalance** for §9.
The 15 classes cover COICOP divisions 01 (food, non-alcoholic beverages) and 02 (alcohol).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py          # rebuilds data/processed/* from data/raw/* (deterministic)
jupyter lab notebooks/coicop_classification_examples.ipynb
```

The committed notebook is **fully executed**; outputs (including all tables/plots) are in
the file. The zero-shot section downloads `facebook/bart-large-mnli` (~1.6 GB) on first
run and is skipped gracefully if `torch`/`transformers` are not installed.

## Method scoreboard (February 2025 hold-out, macro over classes with support ≥ 20)

See `data/processed/method_scoreboard.csv` (written by the notebook) and §11 for the
chart. Highlights from the executed run:

- Attribute rules on *pure* categories: **~100% precision** where purity ≥ 0.95, but low coverage
- Keyword refinement lifts macro-F1 on the heterogeneous vegetable block
- Supervised word+char TF-IDF ML: **best coverage and macro-F1** overall
- Zero-shot BART-MNLI: works with **zero training labels** — useful cold start, behind supervised

## Repository layout

```
├── notebooks/coicop_classification_examples.ipynb   # the single main deliverable (executed)
├── src/prepare_data.py                              # deterministic data pipeline
├── src/build_notebook.py                            # notebook source of truth (rebuilds .ipynb)
├── data/raw/                                        # Zenodo manual labels + UNSD COICOP structure
├── data/processed/                                  # model-ready outputs (committed)
└── docs/sources/                                    # per-source provenance notes
```

## Related work

- The Task Team's own [classification_methods repository](https://github.com/UN-Task-Team-for-Scanner-Data/classification_methods)
  is the official companion for the handbook chapter (notebooks still being published at
  the time of writing); this repo is an independent, community example.

## License

- **Code**: MIT (see [LICENSE](LICENSE)).
- **Derived data** (`data/processed/*`): inherits **CC-BY-SA 4.0** from the Zenodo source dataset.
- **COICOP 2018 structure**: © United Nations, reproduced from the UNSD download above.

## Citation

If you use this material, please cite the UN-CEBD handbook, the Zenodo dataset, and this
repository (see [CITATION.cff](CITATION.cff)).
