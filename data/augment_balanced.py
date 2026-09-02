import numpy as np

from data.augment import SubAnomaly as _BaseSubAnomaly


class SubAnomaly(_BaseSubAnomaly):
    """
    Drop-in replacement for data.augment.SubAnomaly that injects exactly ONE
    anomaly per window and keeps the anomaly types equally represented at
    dataset level.

    Motivation: a single window can only host a small number of anomalous
    timesteps for short anomaly types (a spike covers 2 timesteps, a scale
    burst 5), while long anomaly types (frequency, trend, shapelet) easily
    cover tens of timesteps in the same window. If every type were injected
    with the same probability per window, long types would dominate the total
    amount of anomalous data and become trivially learnable.

    Balancing strategy here: every window still contains a single anomaly of
    a single type, but windows are assigned types with frequency proportional
    to 1/instance_coverage, i.e. spike windows are generated ~16x more often
    than frequency/trend/shapelet windows (budget 32 / spike_len 2) and scale
    windows ~7x more often (budget 32 / scale_len 5). The type for each call
    is chosen greedily as the type with the least accumulated anomalous
    timestep coverage so far (random tie-breaking), which self-corrects and
    keeps the total timestep coverage per type equal across the dataset
    regardless of varying window lengths.

    Auditing: `last_injections` holds (type, covered_timesteps) of the last
    call; `report()` returns per-type window counts and timestep totals.
    """

    TYPE_NAMES = {0: "frequency", 1: "trend", 2: "spike", 3: "scale", 4: "shapelet"}

    def __init__(self, portion_len=None, budget=0.25,
                 spike_len=2, spike_scale=8.0, scale_len=5, scale_scale=3.0,
                 min_budget=10):
        super().__init__(portion_len, balanced=False, budget=budget,
                         spike_len=spike_len, spike_scale=spike_scale,
                         scale_len=scale_len, scale_scale=scale_scale,
                         max_instances=1, min_budget=min_budget)
        self.coverage = np.zeros(self.NUM_TYPES, dtype=float)
        self.window_counts = np.zeros(self.NUM_TYPES, dtype=int)
        self.last_type = None

    def _instance_coverage(self, budget):
        return np.array([budget, budget, self.spike_len, self.scale_len, budget], dtype=float)

    def _pick_type(self, budget):
        projected = self.coverage + self._instance_coverage(budget)
        candidates = np.flatnonzero(projected <= projected.min() + 1e-9)
        anomaly_type = int(np.random.choice(candidates))
        self.coverage[anomaly_type] += self._instance_coverage(budget)[anomaly_type]
        self.window_counts[anomaly_type] += 1
        return anomaly_type

    def _single_segment(self, anomaly_type, window_len, budget):
        if anomaly_type == self.TREND:
            return [(window_len - budget, budget)]
        if anomaly_type == self.SPIKE:
            seg_len = max(1, min(self.spike_len, window_len // 2))
        elif anomaly_type == self.SCALE:
            seg_len = max(1, min(self.scale_len, window_len // 2))
        else:
            seg_len = budget
        start = int(np.random.randint(0, window_len - seg_len))
        return [(start, seg_len)]

    def report(self):
        return {
            self.TYPE_NAMES[t]: {
                "windows": int(self.window_counts[t]),
                "timesteps": float(self.coverage[t]),
            }
            for t in range(self.NUM_TYPES)
        }

    def __call__(self, X):
        """
        Injects one anomaly of the currently least-represented type into one
        random feature (multivariate) or the series (univariate).
        """
        anomalous_window = X.copy()
        window_len = anomalous_window.shape[0]
        budget = self._budget_len(window_len)

        anomaly_type = self._pick_type(budget)
        segments = self._single_segment(anomaly_type, window_len, budget)

        if anomalous_window.ndim > 1:
            i = int(np.random.randint(0, anomalous_window.shape[1]))
            column = anomalous_window[:, i]
            anomalous_window[:, i] = self._apply_type(column, anomaly_type, segments)
        else:
            anomalous_window = self._apply_type(anomalous_window, anomaly_type, segments)

        self.last_injections = [(anomaly_type, sum(l for _, l in segments))]
        self.last_type = anomaly_type
        return anomalous_window
