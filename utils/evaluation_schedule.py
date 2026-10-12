"""Shared epoch cadence for AE/VAE and LEWM phase-two monitoring."""


def evaluation_interval(config):
    """Top-level setting wins; retain the reconstruction setting as a fallback."""
    interval = int(config.get("eval_every", config.get("stage_c", {}).get("eval_every", 10)))
    if interval < 0:
        raise ValueError("eval_every must be nonnegative (0 disables test monitoring)")
    return interval


def should_evaluate(epoch, epochs, interval):
    """Epoch numbers are one-based, including the final scheduled evaluation."""
    return interval > 0 and (epoch % interval == 0 or epoch == epochs)
