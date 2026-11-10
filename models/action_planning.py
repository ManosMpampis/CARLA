"""Categorical Cross-Entropy Method for a learned finite-action LeWM."""

import torch

from models.actions import DiscreteAction


@torch.no_grad()
def plan_discrete_actions(model, current_input, goal_input, *, horizon,
                          samples=128, elites=16, iterations=5, generator=None):
    """Return the best sampled action sequence and latent goal cost per input."""
    if not isinstance(model.action, DiscreteAction):
        raise ValueError("planning requires a model with a discrete action module")
    if (any(type(value) is not int or value < 1 for value in (horizon, samples, elites, iterations))
            or elites > samples):
        raise ValueError("horizon, samples, elites and iterations must be positive, with elites <= samples")
    if current_input.shape != goal_input.shape or current_input.ndim != 3:
        raise ValueError("current and goal windows must have the same (B, C, W) shape")
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device_type=current_input.device.type, enabled=False):
            initial = model.encode(current_input.float())
            goal = model.encode(goal_input.float())
            batch, channels, tokens = initial.shape
            num_actions = model.action.embedding.num_embeddings
            probabilities = initial.new_full((batch, horizon, num_actions), 1 / num_actions)
            best_cost = initial.new_full((batch,), torch.inf)
            best_plan = torch.zeros(batch, horizon, dtype=torch.long, device=initial.device)
            batch_index = torch.arange(batch, device=initial.device)
            for _ in range(iterations):
                drawn = torch.multinomial(probabilities.flatten(0, 1), samples,
                                           replacement=True, generator=generator)
                plans = drawn.view(batch, horizon, samples).transpose(1, 2)
                state = initial.repeat_interleave(samples, dim=0)
                for step in plans.unbind(dim=-1):
                    state = model.predict_action(state, step.reshape(-1))
                errors = (state.reshape(batch, samples, channels, tokens) - goal[:, None]).abs()
                costs, indices = errors.mean(dim=(2, 3)).topk(elites, largest=False)
                candidates = plans[batch_index[:, None], indices]
                improved = costs[:, 0] < best_cost
                best_plan = torch.where(improved[:, None], candidates[:, 0], best_plan)
                best_cost = torch.minimum(best_cost, costs[:, 0])

                counts = torch.zeros_like(probabilities)
                elite_actions = candidates.transpose(1, 2)
                counts.scatter_add_(2, elite_actions, torch.ones_like(elite_actions, dtype=counts.dtype))
                probabilities = 0.1 * probabilities + 0.9 * counts / elites
            return best_plan, best_cost
    finally:
        model.train(was_training)
