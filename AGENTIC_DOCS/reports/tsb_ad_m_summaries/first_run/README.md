# first_run summary report

Generated from `results/tsb_ad_m/benchmark_runs/first_run/summary.json` and the final metrics linked by that manifest. See ../README.md for the command and metric semantics.

| Split | Listed files | Training >500 | Excluded |
| --- | ---: | ---: | ---: |
| all | 200 | 175 | 25 |
| tuning | 20 | 18 | 2 |
| eval | 180 | 157 | 23 |

Seven experiments are included. The encoder, both reconstruction variants, observed_features, and observed_query each completed on 167/175 eligible files (17/18 tuning, 150/157 eval). The observed_features_pos_enc and observed_query_pos_enc variants each failed on all 175 eligible files; their metric cells are blank, with valid-value counts of zero. Failures remain listed in `failed_runs.csv`.

Every completed eligible run has all 14 requested metric values in each of the three modes. There are no missing benchmark entries among eligible files. In total there are 835 completed and 390 failed eligible runs. All successful experiments share the same completed file set.

Verification: `./venv/bin/python tests/verify_tsb_ad_summaries.py` passed, covering strict training filtering, split membership, macro means/sample std, failed and absent jobs, missing/nonfinite values, and duplicate rejection. Generated exports were checked for complete per-metric counts matching completed runs in all nine summaries.
