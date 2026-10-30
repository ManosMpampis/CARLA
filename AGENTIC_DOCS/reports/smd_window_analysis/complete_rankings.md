# Complete SMD window rankings and experiment limitations

All values are equal-machine means over the same 28 SMD machines, validation-selected checkpoints, seed 4. CA = observed cross attention. Except AE/VAE, keys below share the `lewm_encoder/` prefix. F1 is unadjusted. Ranking is descriptive, not a final test or a combined metric.

Late F1 SD is in F1 percentage points; threshold CV is a percentage. These are averages of within-machine statistics, not uncertainty of the mean. Different scored-span groups are different tasks.

## Scored span 1024; input 1024; stride 10

| Experiment | Cal F1 | Oracle F1 | VUS-PR | VUS-ROC | Late F1 SD | Threshold CV | Cal→oracle gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frequency | 0.5201 | 0.7490 | 0.7039 | 0.7028 | 6.22 | 8.76% | 0.2289 |
| frequency_aux | 0.5189 | 0.7562 | 0.7060 | 0.7083 | 5.72 | 10.41% | 0.2373 |
| frequency / recon / default | 0.5125 | 0.7264 | 0.6718 | 0.6755 | 0.82 | 1.72% | 0.2139 |
| ae/default | 0.5108 | 0.7308 | 0.6754 | 0.6759 | 0.72 | 2.23% | 0.2200 |
| frequency_aux_no_film | 0.5082 | 0.7622 | 0.7105 | 0.7066 | 5.25 | 11.65% | 0.2539 |
| frequency_aux / recon / auxilary_added | 0.5053 | 0.7300 | 0.6759 | 0.6872 | 0.39 | 1.20% | 0.2247 |
| time_aux_no_film / recon / default | 0.5027 | 0.7227 | 0.6744 | 0.6842 | 0.71 | 1.56% | 0.2200 |
| time_aux_no_film / recon / auxilary_added | 0.5027 | 0.7227 | 0.6744 | 0.6842 | 0.71 | 1.56% | 0.2200 |
| time / recon / default | 0.5025 | 0.7227 | 0.6705 | 0.6610 | 0.87 | 1.87% | 0.2201 |
| vae/default | 0.5002 | 0.7274 | 0.6708 | 0.6667 | 0.58 | 2.76% | 0.2272 |
| frequency_aux_no_film / recon / auxilary_added | 0.4960 | 0.7317 | 0.6835 | 0.6899 | 0.49 | 1.43% | 0.2357 |
| frequency_aux_no_film / recon / default | 0.4960 | 0.7317 | 0.6835 | 0.6899 | 0.49 | 1.43% | 0.2357 |
| frequency_aux / recon / default | 0.4954 | 0.7280 | 0.6767 | 0.6885 | 0.41 | 1.11% | 0.2326 |
| time_aux / recon / auxilary_added | 0.4899 | 0.7290 | 0.6837 | 0.6884 | 0.72 | 1.78% | 0.2391 |
| time_aux / recon / default | 0.4886 | 0.7318 | 0.6855 | 0.6878 | 0.81 | 1.92% | 0.2433 |
| time_aux | 0.4840 | 0.7414 | 0.6904 | 0.6787 | 4.67 | 13.94% | 0.2574 |
| time_aux_no_film | 0.4566 | 0.7249 | 0.6732 | 0.6603 | 5.18 | 13.93% | 0.2683 |
| time | 0.3476 | 0.6675 | 0.5603 | 0.5764 | 5.30 | 12.14% | 0.3199 |

**frequency:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.1953; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**frequency_aux:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**frequency / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1358; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 20/28.

**ae/default:** Strong simple stable control; flexible reconstruction and MSE-versus-L1 geometry do not directly optimize anomaly separation. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.1328; exact oracle threshold inside clean range on 6/28; near-oracle quantile recovery provably impossible on 17/28.

**frequency_aux_no_film:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 5/28; near-oracle quantile recovery provably impossible on 20/28.

**frequency_aux / recon / auxilary_added:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.1322; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**time_aux_no_film / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Its default/auxilary_added pair has identical saved metrics and is not an independent scoring gain. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.1341; exact oracle threshold inside clean range on 8/28; near-oracle quantile recovery provably impossible on 18/28.

**time_aux_no_film / recon / auxilary_added:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Its default/auxilary_added pair has identical saved metrics and is not an independent scoring gain. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.1341; exact oracle threshold inside clean range on 8/28; near-oracle quantile recovery provably impossible on 18/28.

**time / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1342; exact oracle threshold inside clean range on 6/28; near-oracle quantile recovery provably impossible on 19/28.

**vae/default:** Stable, but no aggregate gain over AE; KL regularization changes the objective and sampled training differs from mean inference. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1216; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**frequency_aux_no_film / recon / auxilary_added:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Its default/auxilary_added pair has identical saved metrics and is not an independent scoring gain. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1341; exact oracle threshold inside clean range on 6/28; near-oracle quantile recovery provably impossible on 18/28.

**frequency_aux_no_film / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Its default/auxilary_added pair has identical saved metrics and is not an independent scoring gain. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1341; exact oracle threshold inside clean range on 6/28; near-oracle quantile recovery provably impossible on 18/28.

**frequency_aux / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1082; exact oracle threshold inside clean range on 8/28; near-oracle quantile recovery provably impossible on 18/28.

**time_aux / recon / auxilary_added:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 14/28 machines; worst-machine calibrated F1 0.1234; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**time_aux / recon / default:** Stable restoration proxy; frozen features and the clean-tail threshold still limit detection. Suite scoring uses the full output, regardless of aux_crop-style configuration names. Calibrated F1 beats always-alert F1 on 14/28 machines; worst-machine calibrated F1 0.1242; exact oracle threshold inside clean range on 7/28; near-oracle quantile recovery provably impossible on 18/28.

**time_aux:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1409; exact oracle threshold inside clean range on 4/28; near-oracle quantile recovery provably impossible on 17/28.

**time_aux_no_film:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1381; exact oracle threshold inside clean range on 6/28; near-oracle quantile recovery provably impossible on 18/28.

**time:** KL-to-L1 geometry, inferred-versus-teacher-forced masks, early validation selection and noisy late thresholds need investigation. Particularly weak calibrated F1; prioritize as a diagnostic negative control rather than a leading detector. Calibrated F1 beats always-alert F1 on 8/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 11/28; near-oracle quantile recovery provably impossible on 11/28.

## Scored span 512; input 1024; stride 10

| Experiment | Cal F1 | Oracle F1 | VUS-PR | VUS-ROC | Late F1 SD | Threshold CV | Cal→oracle gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frequency_aux_no_film / CA / observed_query | 0.4207 | 0.6363 | 0.5642 | 0.7013 | 0.56 | 3.60% | 0.2157 |
| frequency / CA / observed_query | 0.4194 | 0.6291 | 0.5399 | 0.6767 | 0.39 | 2.93% | 0.2098 |
| frequency_aux / CA / observed_query | 0.4150 | 0.6404 | 0.5692 | 0.7059 | 0.59 | 3.16% | 0.2254 |
| time_aux / CA / observed_query | 0.4095 | 0.6300 | 0.5389 | 0.6671 | 0.98 | 4.26% | 0.2205 |
| frequency_aux_no_film / CA / observed_features | 0.4060 | 0.6564 | 0.5647 | 0.6984 | 0.12 | 0.75% | 0.2503 |
| frequency_aux / CA / observed_features | 0.4046 | 0.6489 | 0.5655 | 0.7024 | 0.13 | 0.53% | 0.2443 |
| time_aux / CA / observed_features | 0.3996 | 0.6381 | 0.5408 | 0.6673 | 0.38 | 2.18% | 0.2385 |
| time / CA / observed_query | 0.3986 | 0.5801 | 0.5025 | 0.6426 | 0.79 | 3.15% | 0.1814 |
| frequency / CA / observed_features | 0.3939 | 0.6296 | 0.5392 | 0.6743 | 0.68 | 3.62% | 0.2356 |
| time_aux_no_film / CA / observed_features | 0.3857 | 0.6128 | 0.5307 | 0.6709 | 0.47 | 1.64% | 0.2271 |
| time_aux_no_film / CA / observed_query | 0.3827 | 0.6100 | 0.5328 | 0.6721 | 0.61 | 3.83% | 0.2273 |
| time_aux_no_film / CA / observed_input | 0.3731 | 0.5865 | 0.4997 | 0.6526 | 1.92 | 4.21% | 0.2134 |
| time_aux / CA / observed_input | 0.3607 | 0.5824 | 0.4984 | 0.6443 | 1.70 | 3.43% | 0.2217 |
| time / CA / observed_input | 0.3601 | 0.5961 | 0.5123 | 0.6723 | 1.33 | 3.92% | 0.2360 |
| frequency_aux_no_film / CA / observed_input | 0.3541 | 0.5922 | 0.5096 | 0.6504 | 1.49 | 4.02% | 0.2381 |
| frequency / CA / observed_input | 0.3528 | 0.5888 | 0.5044 | 0.6497 | 1.24 | 3.65% | 0.2359 |
| frequency_aux / CA / observed_input | 0.3450 | 0.5852 | 0.4980 | 0.6512 | 1.46 | 3.99% | 0.2402 |
| time / CA / observed_features | 0.3146 | 0.5615 | 0.4599 | 0.6209 | 1.39 | 4.05% | 0.2469 |

**frequency_aux_no_film / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.0314; exact oracle threshold inside clean range on 9/28; near-oracle quantile recovery provably impossible on 12/28.

**frequency / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.1050; exact oracle threshold inside clean range on 10/28; near-oracle quantile recovery provably impossible on 14/28.

**frequency_aux / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 19/28 machines; worst-machine calibrated F1 0.0934; exact oracle threshold inside clean range on 11/28; near-oracle quantile recovery provably impossible on 12/28.

**time_aux / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 20/28 machines; worst-machine calibrated F1 0.0958; exact oracle threshold inside clean range on 15/28; near-oracle quantile recovery provably impossible on 11/28.

**frequency_aux_no_film / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 19/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 11/28; near-oracle quantile recovery provably impossible on 13/28.

**frequency_aux / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 19/28 machines; worst-machine calibrated F1 0.0746; exact oracle threshold inside clean range on 12/28; near-oracle quantile recovery provably impossible on 12/28.

**time_aux / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.0940; exact oracle threshold inside clean range on 13/28; near-oracle quantile recovery provably impossible on 13/28.

**time / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 15/28 machines; worst-machine calibrated F1 0.1125; exact oracle threshold inside clean range on 15/28; near-oracle quantile recovery provably impossible on 11/28.

**frequency / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.0965; exact oracle threshold inside clean range on 11/28; near-oracle quantile recovery provably impossible on 13/28.

**time_aux_no_film / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 17/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 13/28; near-oracle quantile recovery provably impossible on 13/28.

**time_aux_no_film / CA / observed_query:** Target representation is trainable; inspect whether loss reduction preserves real-fault separation. Calibrated F1 beats always-alert F1 on 18/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 9/28; near-oracle quantile recovery provably impossible on 15/28.

**time_aux_no_film / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.1263; exact oracle threshold inside clean range on 12/28; near-oracle quantile recovery provably impossible on 13/28.

**time_aux / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 15/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 12/28; near-oracle quantile recovery provably impossible on 12/28.

**time / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 16/28 machines; worst-machine calibrated F1 0.0091; exact oracle threshold inside clean range on 14/28; near-oracle quantile recovery provably impossible on 12/28.

**frequency_aux_no_film / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 13/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 15/28; near-oracle quantile recovery provably impossible on 11/28.

**frequency / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 13/28 machines; worst-machine calibrated F1 0.0754; exact oracle threshold inside clean range on 13/28; near-oracle quantile recovery provably impossible on 12/28.

**frequency_aux / CA / observed_input:** Raw-target restoration has the weakest calibrated means among these observed heads; inspect residual dilution and task difficulty. Calibrated F1 beats always-alert F1 on 14/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 14/28; near-oracle quantile recovery provably impossible on 13/28.

**time / CA / observed_features:** Frozen feature target may miss faults discarded by phase one. Calibrated F1 beats always-alert F1 on 12/28 machines; worst-machine calibrated F1 0.0000; exact oracle threshold inside clean range on 13/28; near-oracle quantile recovery provably impossible on 9/28.

## Incomplete newer variants

frequency_aux observed_features_pos_enc has 5 machines and observed_query_pos_enc has 6. Both use input 768 / scored target 256. They are excluded from all complete-family rankings; their subset means cannot be compared with the 28-machine 1024/512 means.

[Per-machine source rows and paths](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/selected_runs.csv)