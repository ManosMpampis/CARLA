import matplotlib.pyplot as plt
import numpy as np

def make_figures(logger, inputs, labels, predictions, mode="Combined", epoch=0):
        """
        inputs: np.ndarray of shape (N, 38)
        labels: np.ndarray of shape (N,)
        predictions: np.ndarray of shape (N,)
        mode: str, either "Train" or "Combined"
        epoch: int
        """
        n_samples, n_features = inputs.shape
        x = np.arange(n_samples)

        for i in range(n_features):
            fig, ax = plt.subplots(figsize=(12, 3))

            y = inputs[:, i]
            ax.plot(x, y, color="black", linewidth=1, label=f"feature {i}")

            # Shade where labels == 1
            label_mask = labels.astype(bool)
            ax.fill_between(
                x,
                y.min()-1,
                (y.max()-y.min()+1)/2,
                where=label_mask,
                color="red",
                alpha=0.2,
                step="mid",
                label="label=1",
            )

            # Shade where predictions == 1
            pred_mask = predictions.astype(bool)
            ax.fill_between(
                x,
                (y.max()-y.min()-1)/2,
                y.max()+1,
                where=pred_mask,
                color="blue",
                alpha=0.2,
                step="mid",
                label="prediction=1",
            )

            ax.set_title(f"Feature {i}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")

            logger.add_figure(f"{mode}/feature_{i}", fig, step=epoch)
            plt.close(fig)