# LeWM phase-one experiments

LeWM is the world-model framework used by these experiments. The predictor
domain and time annotation steering are independent experiment choices. The
auxiliary head supplies both mask annotations and FiLM features, so time
annotation steering is enabled only in the two experiments with that head.

| Prediction domain | Auxiliary head | Time annotation steering | Config |
| --- | --- | --- | --- |
| Frequency (STFT neck) | Yes | Yes | `phase1_frequency_predictor_time_annotation_steering.yml` |
| Time (1D convolution neck) | Yes | Yes | `phase1_time_predictor_time_annotation_steering.yml` |
| Frequency (STFT neck) | No | No | `phase1_frequency_predictor.yml` |
| Time (1D convolution neck) | No | No | `phase1_time_predictor.yml` |

Each config has a distinct `tag_jepa`, so all four can use the same `--version`
without sharing a checkpoint directory. Run one with:

```bash
./venv/bin/python lewm.py --config_env configs/env.yml \
    --config_exp configs/jepa/steering/phase1_time_predictor_time_annotation_steering.yml \
    --fname machine-1-1.txt --version phase1_comparison
```

Without an auxiliary head, training still uses the injected mask to teach the
predictor which latent positions to reconstruct. Validation and scoring use an
empty action because no auxiliary head is available to propose one.
