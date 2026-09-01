# CARLA TSAD

Time-series anomaly detection research. The main line is a convolutional-pyramid JEPA; a novel dual-stream (time + frequency) architecture is under design alongside it.

## Language

### Architecture concepts

**Dual-stream architecture**:
The anomaly detector under design: two parallel analysis pathways over the same input window — a time pathway and a frequency pathway — joined by steering and part-vs-whole comparison.
_Avoid_: two-branch model, hybrid model

**Time pathway**:
The stream that processes the raw time series directly. Primary carrier of localization: where in the window something is strange.
_Avoid_: time branch, temporal stream

**Frequency pathway**:
The stream that processes a frequency-domain representation of the input. Carrier of periodic and spectral structure that the time pathway cannot see.
_Avoid_: frequency branch, spectral stream

**Steering**:
Conditioning of the frequency pathway by the time pathway, so that time-domain evidence decides what and where the frequency pathway analyzes. Soft and differentiable; one-directional (time → frequency) by default.
_Avoid_: gating (too generic), cross-stream conditioning (the research-doc survey term)

**Part-vs-whole comparison**:
The human-inspired checking pattern the model replicates: (i) each sub-segment against a global summary of the whole input, (ii) fine-scale features against coarse-scale features over the same time range, (iii) each sub-segment against the other sub-segments of the same input.
_Avoid_: hierarchical contrasting (TS2Vec's term for a narrower idea), multi-scale loss

**Trunk**:
The shared body of the dual-stream architecture — feature extractors, steering, and bottleneck — to which all pluggable heads attach. Held fixed during a head tournament.
_Avoid_: backbone (means only the feature extractor), encoder (JEPA-line term)

**Pluggable head**:
One of several interchangeable output heads sharing the same trunk; heads are selected empirically, not by commitment.
_Avoid_: decoder, output layer

**Head tournament**:
The experimental plan that swaps pluggable heads and loss tactics on a fixed trunk under a fixed evaluation protocol to pick the scoring tactic.
_Avoid_: ablation (tournaments compare output families, ablations remove parts)

**TF grid**:
The two-dimensional time × frequency representation produced by per-block FFT of a window; the frequency pathway's working surface.
_Avoid_: spectrogram (signal-processing term; here it is a learned feature surface, not an image)

**Mismatch map**:
A per-position map measuring disagreement along one part-vs-whole axis (part↔global, fine↔coarse, or part↔part). Mismatch maps are the trunk's score-bearing outputs.
_Avoid_: error map (implies a predictive operator only), attention map

### Output concepts

**Per-timestep score map**:
Output format A: one anomaly score per input timestep. The format the frozen metrics stack consumes.
_Avoid_: point scores, dense scores

**Interval detection**:
Output format B: (start, length, confidence) boxes on the time axis, YOLO-style.
_Avoid_: bounding boxes, event proposals

### Supervision concepts

**Normal-only training**:
The supervision regime: training and calibration data are assumed nominal; thresholds are derived without test labels.
_Avoid_: semi-supervised (ambiguous), one-class (a specific method family)
