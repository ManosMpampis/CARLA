# LeWM phase-one experiments

LeWM is the world-model framework used by these experiments. The predictor
domain and time annotation steering are independent experiment choices. The
auxiliary head supplies both mask annotations and FiLM features, so time
annotation steering is enabled only in the two experiments with that head.

| Prediction domain | Auxiliary head | Time annotation steering | Config |
| --- | --- | --- | --- |
| Frequency (STFT neck) | Yes | Yes | `frequency_aux/phase1.yml` |
| Time (1D convolution neck) | Yes | Yes | `time_aux/phase1.yml` |
| Frequency (STFT neck) | No | No | `frequency/phase1.yml` |
| Time (1D convolution neck) | No | No | `time/phase1.yml` |
| Frequency (STFT neck) | Yes | No | `frequency_aux_no_film/phase1.yml` |
| Time (1D convolution neck) | Yes | No | `time_aux_no_film/phase1.yml` |

Each config has a distinct `phase1_experiment` parent directory, so all six
can use the same `--version` without sharing a checkpoint directory. Run one with:

```bash
./venv/bin/python lewm.py --config_env configs/env.yml \
    --config_exp configs/lewm_encoder/time_aux/phase1.yml \
    --fname machine-1-1.txt --version phase1_comparison
```

Without an auxiliary head, training still uses the injected mask to teach the
predictor which latent positions to reconstruct. Validation and scoring use an
empty action because no auxiliary head is available to propose one.
