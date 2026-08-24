from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class GradientMonitor:
    def __init__(
        self,
        model: nn.Module,
        logger,
        log_interval: int = 10,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e3,
        log_histograms: bool = True,
        step: int = 0,
        aggregate: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.log_interval = log_interval
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.log_histograms = log_histograms
        self.step_count = step
        self.aggregate = aggregate
        self._gradient_sums = {}
        self._ratio_sums = {}
        self._total_norm_sum = 0.0
        self._aggregate_steps = 0

    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        self.step_count += 1
        total_norm_sq = 0.0
        names: List[str] = []
        norms: List[float] = []
        metrics: Dict[str, float] = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.norm(2).item()
            param_norm = param.norm(2).item()
            total_norm_sq += grad_norm ** 2

            tag = name.replace(".", "/")
            metrics[f"grad_norm/{tag}"] = grad_norm
            ratio = grad_norm / (param_norm + 1e-12)
            metrics[f"update_ratio/{tag}"] = ratio

            if self.aggregate:
                self._gradient_sums[name] = self._gradient_sums.get(name, 0.0) + grad_norm
                self._ratio_sums[name] = self._ratio_sums.get(name, 0.0) + ratio
            else:
                self.logger.scalar_summary("grad_norm", tag, grad_norm, self.step_count)
                self.logger.scalar_summary("update_ratio", tag, ratio, self.step_count)

            if self.log_histograms and not self.aggregate:
                try:
                    self.logger.add_histogram("grad_values", tag, param.grad, self.step_count)
                except ValueError:
                    pass
            names.append(name)
            norms.append(grad_norm)

        total_norm = total_norm_sq ** 0.5
        metrics["grad_norm/total"] = total_norm

        if self.aggregate:
            self._total_norm_sum += total_norm
            self._aggregate_steps += 1
            if self.step_count % self.log_interval != 0:
                return metrics

            count = self._aggregate_steps
            names = sorted(self._gradient_sums)
            norms = []
            for name in names:
                tag = name.replace(".", "/")
                average_norm = self._gradient_sums[name] / count
                average_ratio = self._ratio_sums[name] / count
                self.logger.scalar_summary("grad_norm", tag, average_norm, self.step_count)
                self.logger.scalar_summary("update_ratio", tag, average_ratio, self.step_count)
                norms.append(average_norm)
            total_norm = self._total_norm_sum / count
            self.logger.scalar_summary("grad_norm", "total", total_norm, self.step_count)
            self._gradient_sums.clear()
            self._ratio_sums.clear()
            self._total_norm_sum = 0.0
            self._aggregate_steps = 0
        else:
            self.logger.scalar_summary("grad_norm", "total", total_norm, self.step_count)

            # Push a gradient-flow bar chart as an image every N steps.
            if self.step_count % self.log_interval == 0:
                fig = self._plot_gradient_flow(names, norms)
                self.logger.add_figure("gradient_flow", fig, self.step_count)
                plt.close(fig)

        return metrics

    def _plot_gradient_flow(self, names: List[str], norms: List[float]):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))

        norms_arr = np.array(norms, dtype=float)

        zero_mask = norms_arr == 0.0
        nan_mask = ~np.isfinite(norms_arr)
        vanishing_mask = (~zero_mask) & (~nan_mask) & (norms_arr < self.vanishing_threshold)
        exploding_mask = (~zero_mask) & (~nan_mask) & (norms_arr > self.exploding_threshold)

        floor = 1e-10
        plot_vals = norms_arr.copy()
        plot_vals[zero_mask | nan_mask] = floor

        colors = []
        for is_zero, is_nan, is_vanishing, is_exploding in zip(
                zero_mask, nan_mask, vanishing_mask, exploding_mask):
            if is_zero or is_nan:
                colors.append("crimson")      # dead / zero
            elif is_vanishing:
                colors.append("orange")
            elif is_exploding:
                colors.append("green")
            else:
                colors.append("steelblue")    # healthy

        bars = ax.bar(
            range(len(names)),
            plot_vals,
            color=colors,
            edgecolor="black",
            linewidth=0.3,
        )

        for bar, is_zero, is_nan in zip(bars, zero_mask, nan_mask):
            if is_zero or is_nan:
                label = "0" if is_zero else "NaN"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="crimson",
                    fontweight="bold",
                )

        if np.max(plot_vals) <= floor:
            ax.set_yscale("linear")
            ax.set_ylim(-0.1, 1.0)
            ax.text(0.5, 0.5, "All gradients are zero or non-finite",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="red",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        else:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor * 0.1)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [n.replace(".weight", "w").replace(".bias", "b") for n in names],
            rotation=45, ha="right", fontsize=6,
        )
        ax.set_ylabel("Loss Gradient Norm")
        ax.set_title("Gradient Flow")

        y_min, y_max = ax.get_ylim()
        if self.vanishing_threshold >= y_min:
            ax.axhline(self.vanishing_threshold, color="orange",
                       linestyle="--", linewidth=1)
        if self.exploding_threshold <= y_max:
            ax.axhline(self.exploding_threshold, color="purple",
                       linestyle="--", linewidth=1)

        legend_elements = [
            Patch(facecolor="steelblue", label="healthy"),
            Patch(facecolor="orange", label="vanishing"),
            Patch(facecolor="green", label="exploding"),
            Patch(facecolor="crimson", label="zero / dead"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

        plt.tight_layout()
        return fig
