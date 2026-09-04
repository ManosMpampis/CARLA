# DIAGRAM: JEPA TSAD model & training framework

Mermaid diagrams of the model and training framework as built. Companion to
`DESIGN_TSAD_JEPA.md` (spec) — drawn from code: `carla_jepa.py`,
`models/jepa_core.py`, `models/jepa_pyramid.py`, `models/ema.py`,
`models/codebook.py`, `losses/jepa_losses.py`, `losses/sigreg.py`,
`data/jepa_dataset.py`, `utils/{trainer,masking,scoring}.py`.

---

## 1. Training framework: stages, wiring, artifacts

```mermaid
flowchart TD
    ENV["configs/env.yml<br/>root_dir = results/"] --> CFG["create_config → p<br/>set_seed(seed=4)"]
    EXP["configs/jepa/&lt;arm&gt;.yml<br/>stage · backbone · predictor · horizons<br/>anti_collapse · criterion_kwargs · amp"] --> CFG

    CFG --> DISP{"p['stage']"}

    DISP -- "pretrain (alias pretext)<br/>STAGE A" --> PRE["run_pretrain<br/>latent-prediction pretraining<br/>corpus: single | joint<br/>optional block masking (J2)"]
    DISP -- "adapt<br/>STAGE B" --> ADP["run_adapt<br/>load pretrained_from (strict)<br/>mode: frozen → encoder frozen,<br/>optimizer rebuilt after freeze | finetune"]
    DISP -- "score" --> SCO["run_score<br/>fp32 eval only:<br/>calibrate on clean-train, then evaluate test"]

    RUN[("results/&lt;dataset&gt;/&lt;version&gt;/&lt;fname&gt;/jepa*/<br/>checkpoint.pth.tar (resume state)<br/>model.pth.tar (best val loss)")]
    PRE --> RUN
    ADP --> RUN
    PRE -. "auto-resume when checkpoint exists" .-> PRE

    RUN -- "weights" --> SCO
    SCO --> CAL[("calibration.json<br/>per-channel thresholds · fusion weights · fallback flag")]
    SCO --> NPZ[("scores.npz<br/>fused scores · start/end idxs · cover counts<br/>channels · pred/gt labels")]
    SCO --> MET[("metrics.json")]

    MET --> HON["honest headline:<br/>point AUROC/AP · window AUROC/AP<br/>point F1 without PA · precision/recall · MCC"]
    MET --> PAC["point_adjust_comparability<br/>(PA reported separately)"]
    MET --> NTB["no_training_baseline<br/>(untrained same-architecture model,<br/>identical scoring path)"]

    TB["Logger / TensorBoard:<br/>model graph trace · per-batch losses<br/>latent_var (collapse watch) · val pred_loss"]
    PRE -.-> TB
    ADP -.-> TB
    SCO -.-> TB
```

## 2. Model: JEPAModel (facade over registries)

Wiring is registry-only via `get_jepa_model`: backbone registry
(`resnet_ts` legacy | `jepa_pyramid` | `jepa_transformer`), predictor registry
(`tcn | gru`), anti-collapse registry (`none | sigreg | ema | codebook`).

```mermaid
flowchart TD
    X["window x : (B, C=38, W=256)<br/><i>dataset emits {'ts': (W, C)}; Trainer transposes to (B, C, W)</i>"]

    X --> ENC
    X -.-> TE

    subgraph ENC["PyramidEncoder  (online branch, trainable)"]
        direction TB
        STEM["stem StridedConvBlock k=7 s=1<br/>conv → BN → GELU → dropout"] --> Z0
        Z0["L0 latents (B, 32, 256) · stride 1 · sub-window = 1 step<br/><i>finest detail</i>"] --> PL1["PyramidLevel L1:<br/>downsample ÷2 + refine"]
        PL1 --> Z1["L1 (B, 32, 128) · s2 · sub-window = 2 steps"]
        Z1 --> PL2["PyramidLevel L2:<br/>downsample ÷2 + refine"]
        PL2 --> Z2["L2 (B, 64, 64) · s4 · sub-window = 4 steps"]
        Z2 --> PL3["PyramidLevel L3:<br/>downsample ÷2 + refine"]
        PL3 --> Z3["L3 (B, 96, 32) · s8 · sub-window = 8 steps<br/><i>coarsest trend</i>"]
    end

    Z0 & Z1 & Z2 & Z3 --> PRED
    Z0 & Z1 & Z2 & Z3 -.-> SCB

    subgraph PRED["one causal predictor per level  (registry: tcn | gru)"]
        PR0["CausalTCNPredictor ×4 levels<br/>left-padded causal convs, dilations (1,2,4), k=3<br/>1×1 causal head → horizons k∈{1,2}"]
        PRG["alt arm: GRUPredictor<br/>(left-to-right, linear head)"]
        PR0 ~~~ PRG
    end

    PRED --> OUT["predicted per level (B, k, D_l, T_l):<br/>pred at position t+k from strictly-past latents ≤ t"]

    subgraph TGTBRANCH["target branch (stop-gradient by construction)"]
        TE["EMA arm (J3): EMAWrapper teacher<br/>deepcopy of encoder, m=0.99925,<br/>always eval-mode (BN frozen),<br/>updated after every optimizer step"]
        DET["default arms (J1/J2/JT/J4):<br/>targets = latents.detach()"]
    end
    TE --> TG["targets {level: (B, D_l, T_l)}<br/>layer-normed before loss on EMA arm<br/>(criterion target_norm='layer')"]
    DET --> TG

    subgraph CBARM["codebook arm (J4) extras"]
        SCB["SoftCodebook<br/>K=64 prototypes per level, soft-attention routing,<br/>k-means warmup init from first-epoch latents<br/>(first 8 batches); adds quantization loss +<br/>score-time signals: dist-to-prototype, attn entropy"]
    end

    OUT --> LOSS["→ JEPALoss (see §3)"]
    TG --> LOSS
```

## 3. Loss and training loop (Trainer)

```mermaid
flowchart TD
    DL["DataLoader — JEPADataset<br/>sliding windows (wsz=256), train split only;<br/>val split = tail of TRAIN series<br/>(test data unreachable during selection)"]

    DL --> MC{"MaskingCollator?<br/>stage_a.masking.mode == 'block'"}
    MC -- "yes · stage-A masked / J2" --> MASK["contiguous blocks sampled in token space per level<br/>{Lℓ: bool (B, T_ℓ)}; coarse levels stay visible;<br/>≥1 visible token guaranteed"]
    MC -- "no (J1 default)" --> FWD
    MASK --> FWD

    subgraph STEP["one optimization step (Trainer.train_one_epoch)"]
        FWD["autocast bf16 when amp ∧ CUDA (GradScaler)<br/>JEPAModel.forward(x, mask) →<br/>latents · sg-targets · predictions (+codebook term)"]
        CRIT["JEPALoss(outputs)"]
        BP["optimizer.zero_grad →<br/>scaler.scale(loss).backward →<br/>scaler.step/update"]
        EMAU["model.update_ema()<br/>(EMA arm only)"]
        FWD --> CRIT --> BP --> EMAU
    end

    CRIT --> PL["per level ℓ, per horizon k ∈ 1..K:<br/>L1 between pred shifted by k and target<br/>weighted by mask on target positions<br/>(all tokens counted when mask-free)"]
    PL --> MEAN["pred_loss = mean over levels<br/>(optional level_weights)"]
    CRIT --> SG["sigreg = sliced Epps–Pulley statistic per token<br/>(16 fixed random slices, freq grid [0.2, 4])<br/>SIGReg always sees FULL latents · λ≈0.1"]
    CRIT --> CBL["codebook soft-quantization distance · λ_codebook"]
    MEAN --> LOSS["loss = pred_loss<br/>+ λ_sigreg·sigreg<br/>+ λ_codebook·codebook"]
    SG --> LOSS
    CBL --> LOSS

    STEP -. "every 100 batches: log latent_var (collapse watch)" .-> EPOCH
    STEP --> EPOCH

    subgraph EPOCH["epoch boundary · checkpoint selection"]
        SCH["scheduler.step()"]
        VAL["validate(): mean val pred_loss<br/>strictly in eval mode<br/>(BN buffers must not inflate selection)"]
        SEL{"val_loss < best?"}
        SAVB["save model.pth.tar (best weights)"]
        SAVC["save checkpoint.pth.tar<br/>(model+opt+sched+epoch+best_val_loss+stage meta)"]
        SCH --> VAL --> SEL
        SEL -- "improved" --> SAVB --> SAVC
        SEL -- "last epoch" --> SAVC
    end

    RESUME["next run: Trainer.resume() restarts from<br/>checkpoint.pth.tar (start_epoch, best_val_loss)"]
    SAVC -.-> RESUME
```

Frozen-adaptation details (`stage_b.mode: frozen`): encoder params
`requires_grad=False` and kept in `eval()` so BN stats never move;
EMA teacher always runs in eval mode.

## 4. Scoring and calibration pipeline (`run_score`)

```mermaid
flowchart TD
    CLEAN["clean-train series"] --> SS
    TEST["test series + labels<br/><i>labels used ONLY for final metric computation</i>"] --> TS["Scorer.score_series on test series<br/>(same path as below)"]

    subgraph SCOREPATH["Scorer.score_series(series, wsz, stride) — identical path both sides"]
        SS["slide windows (wsz=256, stride)<br/>batches of 256 → JEPAModel.score<br/>(always fp32, eval mode)"]
        TOK["per level ℓ: token error e(t) =<br/>mean over valid horizons k of ‖pred_k(t) − tgt(t+k)‖₁<br/>aligned to future position, horizon-weighted"]
        MAP["map tokens → input timesteps:<br/>repeat_interleave(stride_ℓ)"]
        FUSE["sum across levels ÷ level count<br/>→ fused (B, W)"]
        AGG["overlap-aware aggregation across covering windows:<br/>Σ contributions ÷ cover_count;<br/>tail timesteps forward-filled"]
        CODESIG["codebook arm extra channels:<br/>dist-to-prototype · attention entropy"]
        SS --> TOK --> MAP --> FUSE --> AGG
        FUSE -.-> CODESIG -.-> AGG
        CH["channels: fused + L0..L3 (+ signals)"]
        AGG --> CH
    end

    CH --> CFIT
    PROBE["SubAnomaly injected probes:<br/>200 windows sampled from TRAIN dataset<br/>anomaly severity portion=0.99"] --> PW["Scorer.score_windows →<br/>per-window means<br/>(same statistic both sides:<br/>clean side also windowed means)"]
    PW --> CFIT["Calibrator.fit(clean, probes?)<br/><i>train-side inputs only — no test reachability</i>"]

    CFIT --> THR["per-channel thresholds:<br/>quantile(0.995) of clean-train distribution<br/>fusion weights ∝ probe separation (p̄−c̄)/σ_c<br/>fallback if probes thin/absent: plain mean fusion"]
    THR --> TFUS["fuse(clean-train channels) →<br/>threshold_fused = q(0.995) of fused clean scores"]
    TFUS --> SAVEC[("calibration.json")]
    SAVEC --> BIN

    TS --> CH2["channels (test side)"]
    CH2 --> FUS2["Calibrator.fuse(test channels)<br/>calibrated weights or mean fallback"]
    THR --> FUS2
    FUS2 --> BIN["pred_labels = fused_test ≥ threshold_fused"]
    BIN --> MM["combine_all_evaluation_scores<br/>(frozen metrics stack)"]

    MM --> HON["honest section:<br/>point AUROC/AP · window AUROC/AP<br/>F1 no PA · MCC"]
    MM --> PA["point-adjust section<br/>(literature comparability only)"]
    BASE["no-training baseline:<br/>untrained same-architecture model<br/>through the identical path<br/>(mandatory design guardrail)"] --> MM

    NPZ[("scores.npz")] 
    MM --> NPZ
```

---

## Component ↔ file map

| Diagram element | Code |
|---|---|
| Stage dispatch, pretrain/adapt/score drivers | `carla_jepa.py` |
| Registry wiring (backbone/predictor/anti-collapse → JEPAModel) | `utils/common_config.get_jepa_model`, `models/__init__.BACKBONE_REGISTRY`, `models/jepa_core.PREDICTOR_REGISTRY` |
| Pyramid encoder, StridedConvBlock, TCN/GRU predictors | `models/jepa_pyramid.py` |
| JEPAModel facade (encode/predict/forward/score) | `models/jepa_core.py` |
| EMA teacher | `models/ema.py` |
| Soft codebook | `models/codebook.py` |
| Dense JEPA criterion (+ mask weighting, codebook, SIGReg terms) | `losses/jepa_losses.py` |
| SIGReg (sliced Epps–Pulley) | `losses/sigreg.py` |
| Windows dataset (train-only normalization, train-tail val) | `data/jepa_dataset.py` |
| Block masking collator | `utils/masking.py` |
| Training loop (AMP, EMA update, eval-mode validation, checkpoints/resume) | `utils/trainer.py` |
| Sliding-window scorer (cover-count-aware aggregation) + Calibrator | `utils/scoring.py` |
| Frozen metrics stack | `metrics/metrics.py::combine_all_evaluation_scores` |

Arm matrix (from spec §2): **J1** plain + SIGReg (primary) · **J2** stage-A block-masked pretraining · **J3** EMA teacher · **J4** codebook · **JT** transformer encoder · corpus axis `{single (headline), joint (exploratory)}`.
