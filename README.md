## Files

- `Public_PMV_Element_Data.xlsx`: public chain-level data, complete codebook, expected results, and method notes.
- `reproduce_section_4_1.py`: stand-alone analysis and figure script.
- `requirements.txt`: pinned software dependencies.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_section_4_1.py Public_PMV_Element_Data.xlsx --output-dir reproduced_section_4_1
```

The output directory will contain three appendix tables, a combined P-M-V result file, Figure 3, run metadata, and `verification.json`. 
## Analysis rules

- One row is one narrative unit.
- P and M are single-coded.
- V is multi-label across three columns; the same value counts at most once within a narrative unit.
- The 2 x 2 table uses Policy/Case as rows and element present/absent as columns.
- Use a two-sided Fisher exact test when any expected cell is below 5; otherwise use Pearson chi-square with Yates continuity correction.
- Primary Section 4.1 inference uses a binary logit comparison of Case versus Policy with CR1 standard errors clustered by anonymous source ID.
- Odds ratios above 1 indicate higher odds in case narratives; odds ratios below 1 indicate higher odds in policy narratives.
- P-values are two-sided and unadjusted for multiplicity.
