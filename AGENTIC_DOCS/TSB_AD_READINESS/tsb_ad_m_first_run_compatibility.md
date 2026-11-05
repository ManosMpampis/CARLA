# TSB-AD-M first_run compatibility check

Date: 2026-10-30. This report describes the current seven-experiment selection in
`configs/tsb_ad_m/experiments.yml`, with the user's geometry preserved.

## Original failure and repair

The two positional-encoding config paths used `./lewm_encoder/...`, which resolved
inside `configs/tsb_ad_m/`. Changed only those paths to `../lewm_encoder/...`.
The existing `first_run` summary contained 44 planning failures: 33 for the
positional config path and 11 from an earlier missing inherited manifest path.
All currently selected YAML configs now resolve.

## Full collection dry run

- All 200 CSVs loaded successfully.
- 167 files have valid geometry for the frequency encoder and four non-positional heads.
- 33 files have fewer than 64 validation rows at `val_fraction: 0.1`. The frequency
  encoder is incompatible with these files, so its dependent heads cannot run.
- Both positional-encoding heads are incompatible with all 200 files:
  `target_crop: [513, 769]` exceeds the requested 512-row window (which may shrink).
- 433 experiment/file pairs have direct geometry errors, before counting heads
  blocked by incompatible encoders.
- No geometry settings, validation fractions, model selections, or existing run
  artifacts were changed by this check.

Planning now validates cross-attention crops with the same geometry resolver used
by the model. Dry runs report invalid explicit crops before any training begins.
Normal runs retain the existing failure policy: record failures, continue independent
experiments, and exit nonzero if any run fails. `--fail-fast` stops at the first failure.
A nonzero dry-run exit is therefore expected with the preserved selection.

Full per-file details are in [tsb_ad_m_first_run_compatibility.json](tsb_ad_m_first_run_compatibility.json).

## Verification

`tests/verify_tsb_ad_planning.py` reproduced the missing early crop validation
before the fix and passes afterward. It asserts that the explicit crops,
positional encoding, and validation fraction are unchanged.

A one-epoch CPU smoke run on the first two real files completed the five
compatible experiments on Genesis and reported the two positional crop failures.
The short-prefix MSL file reported the encoder failure and dependent-head
failures. Outputs were isolated under `/tmp/carla-tsb-m-geometry-smoke`; the
existing `results/tsb_ad_m/first_run` artifacts were not changed.

The extracted crop resolver matches the original model method on 35 valid and
invalid geometry cases. `tests/verify_model_builders.py` fails an existing crop
shape expectation; the same failure occurs with the original pre-fix crop method.

## Files whose frequency encoder is incompatible

- `002_MSL_id_1_Sensor_tr_500_1st_900.csv` (prefix 500 rows)
- `004_MSL_id_3_Sensor_tr_530_1st_630.csv` (prefix 530 rows)
- `010_MSL_id_9_Sensor_tr_554_1st_1172.csv` (prefix 554 rows)
- `012_MSL_id_11_Sensor_tr_539_1st_940.csv` (prefix 539 rows)
- `013_MSL_id_12_Sensor_tr_554_1st_1200.csv` (prefix 554 rows)
- `015_MSL_id_14_Sensor_tr_575_1st_1250.csv` (prefix 575 rows)
- `016_MSL_id_15_Sensor_tr_500_1st_780.csv` (prefix 500 rows)
- `017_MSL_id_16_Sensor_tr_512_1st_1850.csv` (prefix 512 rows)
- `075_SMD_id_19_Facility_tr_564_1st_664.csv` (prefix 564 rows)
- `078_SMD_id_22_Facility_tr_500_1st_326.csv` (prefix 500 rows)
- `080_LTDB_id_2_Medical_tr_500_1st_266.csv` (prefix 500 rows)
- `081_LTDB_id_3_Medical_tr_500_1st_26.csv` (prefix 500 rows)
- `086_SVDB_id_3_Medical_tr_533_1st_633.csv` (prefix 533 rows)
- `088_SVDB_id_5_Medical_tr_500_1st_168.csv` (prefix 500 rows)
- `091_SVDB_id_8_Medical_tr_500_1st_126.csv` (prefix 500 rows)
- `093_SVDB_id_10_Medical_tr_500_1st_37.csv` (prefix 500 rows)
- `096_SVDB_id_13_Medical_tr_500_1st_444.csv` (prefix 500 rows)
- `098_SVDB_id_15_Medical_tr_500_1st_303.csv` (prefix 500 rows)
- `105_SVDB_id_22_Medical_tr_500_1st_417.csv` (prefix 500 rows)
- `116_TAO_id_1_Environment_tr_500_1st_3.csv` (prefix 500 rows)
- `117_TAO_id_2_Environment_tr_500_1st_1.csv` (prefix 500 rows)
- `118_TAO_id_3_Environment_tr_500_1st_7.csv` (prefix 500 rows)
- `119_TAO_id_4_Environment_tr_500_1st_1.csv` (prefix 500 rows)
- `120_TAO_id_5_Environment_tr_500_1st_3.csv` (prefix 500 rows)
- `121_TAO_id_6_Environment_tr_500_1st_7.csv` (prefix 500 rows)
- `122_TAO_id_7_Environment_tr_500_1st_19.csv` (prefix 500 rows)
- `123_TAO_id_8_Environment_tr_500_1st_62.csv` (prefix 500 rows)
- `124_TAO_id_9_Environment_tr_500_1st_1.csv` (prefix 500 rows)
- `125_TAO_id_10_Environment_tr_500_1st_9.csv` (prefix 500 rows)
- `126_TAO_id_11_Environment_tr_500_1st_7.csv` (prefix 500 rows)
- `127_TAO_id_12_Environment_tr_500_1st_24.csv` (prefix 500 rows)
- `128_TAO_id_13_Environment_tr_500_1st_7.csv` (prefix 500 rows)
- `137_CreditCard_id_1_Finance_tr_500_1st_541.csv` (prefix 500 rows)
