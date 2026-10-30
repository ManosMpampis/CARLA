# Reference numbers (published SMD/PSM results)

Purpose: record published Server Machine Dataset (SMD) and Pooled Server Metrics (PSM) results we intend to compare against in this repo (CARLA-style pretext + classification TSAD pipeline). Values are transcribed verbatim from the cited sources — never estimated. Provenance flags:

- `[P]` **paper-reported** — number appears in the method's own peer-reviewed paper/preprint (authors' own run).
- `[R:<reproducer>]` **third-party reproduced** — number produced and published by a later paper/benchmark that re-ran the method under the reproducer's protocol.
- `[I]` **informal** — third-party number from a GitHub issue/repo discussion (weakest provenance; use only for sanity checks, never as headline bars).

**Warning — conventions are NOT comparable across rows.** Most pre-2024 rows are *point-adjusted* F1 (`F1_PA`) computed at the *best threshold searched on the test set*: these are known to be strongly inflated (see *Evaluation-convention warnings* below). Rows labelled `no-PA` (plain point-wise F1), `Aff-*` (affiliation, Huet et al.), `AUC-ROC`, `R_AUC_*` / `VUS_*` (range-based, Paparrizos et al.) follow different, mostly stricter protocols. Never mix rows across protocols when setting a target bar. Also note SMD aggregates over 28 machines and macro/micro averaging choices alone shift reported F1 substantially. Harvest date: **2026-08-24** (automated web harvest).

---

## SMD (Server Machine Dataset)

| Method | Year | Metric(s) | Value(s) | Point-adjust? | Provenance | Source |
|---|---|---|---|---|---|---|
| OmniAnomaly | 2019 | P / R / F1 | 0.8334 / 0.9449 / 0.8857 | Yes (PA; POT or best-F1 threshold) | [P] | Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network", KDD 2019, DOI 10.1145/3292500.3330672 |
| OmniAnomaly | 2019 | F1 / F1\* | F1 0.9441, F1\* 0.9620 (P 0.9809, R 0.9438) | Yes (F1\* = F1 from machine-averaged P/R; best-F1 threshold) | [R:USAD'20] | Audibert et al., KDD 2020, DOI 10.1145/3394486.3403392, Table 2 |
| OmniAnomaly | 2019 | P / R / F1 | 83.68% / 86.82% / 85.22% | Yes (PA + best-F1 threshold) | [R:DCdetector'23] | Yang et al., "DCdetector", KDD 2023, arXiv:2306.10347, Table 1 |
| OmniAnomaly | 2019 | F1 (no-PA) | 0.4591 (P 0.3067, R 0.9126; AU-PR 0.365±0.202) | No | [R:CARLA'25] | Darban et al., "CARLA", Pattern Recognition 157:110874, 2025, arXiv:2308.09296, Table 2 |
| USAD | 2020 | P / R / F1 / F1\* | 0.9314 / 0.9617 / 0.9382 / 0.9463 | Yes (PA + best-F1 threshold over test set) | [P] | Audibert et al., KDD 2020, DOI 10.1145/3394486.3403392, Table 2 |
| MTAD-GAT | 2020 | F1 (no-PA) | 0.3473 (P 0.2473, R 0.5834) | No | [R:CARLA'25] | Zhao et al., arXiv:2009.02040 (method); values from Darban et al. 2025, Table 2. Original paper evaluates SMAP/MSL/TSA only — **no canonical author-published SMD number found** |
| GDN | 2021 | F1-PA / F1 (no-PA) | 93.0% / 42.0% | Both listed separately | [I] | Deng & Hooi, AAAI 2021 (DOI 10.1609/aaai.v35i5.16523) evaluate SWaT/WADI only — no original SMD number; values from thuml/Anomaly-Transformer GitHub issue #34 |
| Anomaly Transformer | 2022 | P / R / F1 | 89.40% / 95.45% / 92.33% | Yes (PA, Xu et al. 2018 adjustment; best-F1 threshold) | [P] | Xu et al., ICLR 2022 Spotlight, arXiv:2110.02642, Table 1 |
| Anomaly Transformer | 2022 | F1-PA / F1 (no-PA) | reproduced F1-PA 0.8944 (paper 0.9233); no-PA F1 ≈ 0.02 | Both listed separately | [I] | thuml/Anomaly-Transformer GitHub issue #34 |
| Anomaly Transformer | 2021/22 | F1 (no-PA) | 0.3043 (P 0.2060, R 0.5822) | No | [R:CARLA'25] | Darban et al. 2025, Table 2 |
| TimesNet (anomaly head) | 2023 | F1 | 85.81% (Inception) / 85.12% (ResNeXt backbone) | Yes (follows Anomaly Transformer PA protocol) | [P] | Wu et al., "TimesNet", ICLR 2023, arXiv:2210.02186, Table 5 |
| TimesNet (anomaly head) | 2023 | F1 (no-PA) | 0.3385 (P 0.2450, R 0.5474) | No | [R:CARLA'25] | Darban et al. 2025, Table 2 |
| DCdetector | 2023 | P / R / F1 | 83.59% / 91.10% / 87.18% | Yes (PA + best-F1 threshold) | [P] | Yang et al., KDD 2023, arXiv:2306.10347, Table 1 |
| CATCH | 2025 | Affiliated-F1 / AUC-ROC | Aff-F1 0.847, AUC-ROC 0.811 | No (affiliation + score-based metrics) | [P] | Wu et al., "CATCH", ICLR 2025, arXiv:2410.12261, Table 2 |
| CARLA | 2025 | P / R / F1 / AU-PR | 0.4276 / 0.6362 / 0.5114 / AU-PR 0.507±0.195 | No (explicitly no PA) | [P] | Darban et al., Pattern Recognition 157:110874, Jan 2025, arXiv:2308.09296, Table 2 |
| CARLA | 2025 | F1-PA (appendix) | 0.7515 (Prec_PA 0.6757, Rec_PA 0.8465) | Yes (appendix-only comparison) | [P] | Darban et al. 2025, Appendix B/C |

Additional no-PA reproductions on SMD (all from Darban et al. 2025, Table 2): LSTM-VAE F1 0.2980, THOC 0.1679, TranAD 0.3609, TS2Vec 0.1728, random scoring 0.1731.

## PSM (Pooled Server Metrics)

Dataset introduced by Abdulaal et al., "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization", KDD 2021.

| Method | Year | Metric(s) | Value(s) | Point-adjust? | Provenance | Source |
|---|---|---|---|---|---|---|
| Anomaly Transformer | 2022 | P / R / F1 | 96.69% / 94.07% / 97.89% | Yes (PA + best-F1 threshold) | [P] | Xu et al., ICLR 2022, arXiv:2110.02642, Table 1 |
| Anomaly Transformer | 2022 | F1-PA / F1 (no-PA) | reproduced F1-PA 0.9750 (paper 0.9789); no-PA F1 ≈ 0.02 | Both listed separately | [I] | thuml/Anomaly-Transformer GitHub issue #34 |
| Anomaly Transformer | 2022 | Acc / F1 / Aff-P / Aff-R / R_AUC_ROC / R_AUC_PR / VUS_ROC / VUS_PR | 98.68 / 97.37 / 55.35 / 80.28 / 91.83 / 93.03 / 88.71 / 90.71 (%), all | Multi-metric block (affiliation/range-based, no PA inflation) | [P] (table compiled by DCdetector paper) | Yang et al., KDD 2023, arXiv:2306.10347, Table 2 |
| DCdetector | 2023 | P / R / F1 | 97.14% / 98.74% / 97.94% | Yes (PA + best-F1 threshold) | [P] | Yang et al., KDD 2023, arXiv:2306.10347, Table 1 |
| DCdetector | 2023 | Acc / F1 / Aff-P / Aff-R / R_AUC_ROC / R_AUC_PR / VUS_ROC / VUS_PR | 98.95 / 97.94 / 54.71 / 82.93 / 91.55 / 92.93 / 88.41 / 90.58 (%), all | Multi-metric block (affiliation/range-based, no PA inflation) | [P] | Yang et al., KDD 2023, arXiv:2306.10347, Table 2 |
| TimesNet (anomaly head) | 2023 | F1 | 97.47% (Inception) / 95.21% (ResNeXt) | Yes (follows Anomaly Transformer PA protocol) | [P] | Wu et al., ICLR 2023, arXiv:2210.02186, Table 5 |
| CATCH | 2025 | Affiliated-F1 / AUC-ROC | Aff-F1 0.859, AUC-ROC 0.652 | No (affiliation + score-based metrics) | [P] | Wu et al., ICLR 2025, arXiv:2410.12261, Table 2 |
| OmniAnomaly | 2019 | P / R / F1 | 88.39% / 74.46% / 80.83% | Yes (PA + best-F1 threshold) | [R:DCdetector'23] | Yang et al., KDD 2023, arXiv:2306.10347, Table 1 |
| USAD | 2020 | — | not found (original paper does not evaluate PSM; no verified third-party PSM reproduction located in this harvest) | — | — | — |
| MTAD-GAT | 2020 | — | not found (original paper predates PSM adoption; no verified reproduction located) | — | — | — |
| GDN | 2021 | — | not found (original paper uses SWaT/WADI only) | — | — | — |
| CARLA | 2025 | — | not found (paper's benchmarks are MSL/SMAP/SMD/SWaT/WADI/Yahoo-A1/KPI; PSM not used) | — | — | Darban et al. 2025 |

---

## Recent SOTA snapshot (2023–2026)

Protocol noted per entry; do not rank across protocols.

- **DCdetector** (Yang et al., KDD 2023, arXiv:2306.10347) — dual-attention contrastive. SMD PA-F1 87.18%, PSM PA-F1 97.94%; additionally publishes affiliation + range-based metrics for PSM (VUS_PR 90.58%) but not SMD. Under CARLA's no-PA re-evaluation its SMD F1 collapses to 0.0828 (Darban et al. 2025).
- **TimesNet anomaly head** (Wu et al., ICLR 2023, arXiv:2210.02186) — reconstruction-error head on 2D-variations backbone, PA protocol. SMD 85.81%, PSM 97.47%. CARLA's no-PA re-evaluation: SMD F1 0.3385.
- **CARLA** (Darban et al., Pattern Recognition 157:110874, Jan 2025, arXiv:2308.09296) — the direct ancestor of this repo's two-stage pipeline (ResNet pretext w/ kernel sizes [8,5,3], rep dim 128, window 200). Honest protocol: no PA anywhere in the main tables. SMD no-PA F1 0.5114, AU-PR 0.507±0.195 (best of 11 methods incl. DCdetector/TimesNet in their re-run); appendix F1_PA 0.7515.
- **CATCH** (Wu et al., ICLR 2025, arXiv:2410.12261) — frequency-patching channel-aware reconstruction; honest protocols (Affiliated-F1 + AUC-ROC, no PA). SMD Aff-F1 0.847 / AUC-ROC 0.811; PSM Aff-F1 0.859 / AUC-ROC 0.652.
- Not verifiable in this harvest: FreCT and other candidate 2024–2026 entries were either not locatable with checkable SMD/PSM tables or could not be confirmed; omitted rather than guessed. (Also: "WATS" could not be verified as a TSAD evaluation-benchmark citation and is excluded.)

## Evaluation-convention warnings

Point-adjust (PA) critiques — read before comparing any PA row above:

- **Kim, Choi, Choi, Lee, Yoon, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection", AAAI 2022 (arXiv:2109.05257)** — a *random* anomaly score reaches SOTA-level F1_PA on MSL/SMAP/SWaT/WADI; PA overestimation depends on segment-length/anomaly-ratio distribution (SMD's short segments cap achievable F1_PA near ~0.8, so SMD is less inflated than others). Proposes PA%K mitigation.
- **Liu & Paparrizos, "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" (TSB-AD), NeurIPS 2024 D&B** — identifies VUS-PR as the most reliable measure; a random score ranks 26/33 detectors under PA-F1; simple statistical methods often beat deep models under fair evaluation.
- **Paparrizos et al., "Volume Under the Surface..." (TSB-UAD/VUS), PVLDB 15(8):1697–1711, 2022** — introduces VUS_ROC/VUS_PR and range-based metrics used by this repo; shows threshold-metric sensitivity to labeling lag.
- **Huet et al., "Local Evaluation of Time Series Anomaly Detection Algorithms", KDD 2022** — origin of the affiliation precision/recall convention used by this repo and by DCdetector/CATCH tables.
- **Wang et al., "Nominality Score Conditioned Time Series Anomaly Detection...", NeurIPS 2023 (NPSR)** — shows simple heuristic scores outperform all evaluated deep methods once PA is applied (best-F1-PA) on SWaT/WADI/PSM/MSL; argues F1_PA is unreliable.
- **Empirical collapse demos**: re-running Anomaly Transformer without PA drops SMD/PSM F1 from ~0.92/0.98 to ~0.02 (thuml issue #34); CARLA's appendix shows a random model beating all SOTA methods under PA (arXiv:2308.09296 App. B/C); DCdetector's SMD F1 falls 87.18% → 8.28% under CARLA's no-PA re-run.
- **Thresholding caveat**: nearly every high number above (USAD, Anomaly Transformer, DCdetector, TimesNet, CARLA's own F1/AU-PR) selects the decision threshold by maximizing test-set F1 — an oracle upper bound, not deployable performance.
- **Aggregation caveat (SMD)**: 28 machines × {global-average, macro, micro} aggregation produce materially different F1 for the same runs; Alves et al. 2026 (arXiv:2603.18985) show PCA matches OmniAnomaly once protocols are equalized.

---

*Harvested automatically via web search on 2026-08-24. Every value above was transcribed from the cited source text during this session; none are estimates. Numbers must be re-checked by the repo owner against the primary PDFs before being treated as authoritative target bars.*
