"""Exercise four LeWM phase-one prediction and steering combinations."""
import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models.builders import get_lewm_model  # noqa: E402
from losses.lewm import LeWMLoss  # noqa: E402
from models.lewm import FreqPredictor, TimePredictor  # noqa: E402
from utils.common_config import get_criterion  # noqa: E402


def main():
    torch.manual_seed(4)
    for with_aux in (True, False):
        for domain in ("frequency", "time"):
            suffix = "_time_annotation_steering" if with_aux else ""
            name = f"phase1_{domain}_predictor{suffix}.yml"
            encoder_name = domain + ("_aux" if with_aux else "")
            path = os.path.join(REPO, "configs/lewm_encoder", encoder_name, "phase1.yml")
            with open(path) as stream:
                config = yaml.safe_load(stream)
            assert config["aux_kwargs"]["with_aux"] == with_aux
            assert config["predictor_kwargs"]["domain"] == domain
            assert config["predictor_kwargs"]["time_steering"] == with_aux
            assert (config["criterion_kwargs"]["w_aux"] > 0) == with_aux
            assert domain in config["experiment_description"]
            assert ("time annotation steering" in config["experiment_description"]) == with_aux
            assert config["tag_jepa"] == f"{domain}_predictor{suffix}"
            assert config["backbone"] == "lewm_resnet"
            assert isinstance(get_criterion(config), LeWMLoss)

            config["model_kwargs"].update(in_channels=4, enc_channels=[8, 12],
                                          enc_strides=[1, 1], dropout=0.0)
            config["aux_kwargs"]["aux_channels"] = [8, 8, 8]
            config["predictor_kwargs"].update(stem_channels=8,
                                               neck_widths=[8, 8, 8])
            model = get_lewm_model(config)
            assert isinstance(model.predictor,
                              FreqPredictor if domain == "frequency" else TimePredictor)
            assert (model.aux is not None) == with_aux
            x = torch.randn(2, 4, 64)
            mask = torch.zeros(2, 64)
            mask[:, 8:24] = 1
            injected = x + mask.unsqueeze(1)
            batch_mask = {"X_inj": injected, "input": mask}
            output = model(x, mask=batch_mask)
            assert output["predicted"]["L0"].shape == (2, 12, 64)
            assert (output["mask_logits"] is not None) == with_aux
            assert config["criterion"] == "lewm"
            losses = LeWMLoss(**config["criterion_kwargs"])(output)
            assert torch.isfinite(losses["loss"])
            if not with_aux:
                assert losses["aux_loss"].item() == 0
            losses["loss"].backward()
            assert model.mask_token.grad is not None
            assert any(p.grad is not None for p in model.predictor.parameters())
            if with_aux:
                assert any(p.grad is not None for p in model.aux.parameters())
                assert all(p.grad is not None for p in model.predictor.film0.parameters())

            model.eval()
            with torch.no_grad():
                proposed = model(x, mask=batch_mask)
                scored = model.score(x)
            assert proposed["predicted"]["L0"].shape == (2, 12, 64)
            assert scored["fused"].shape == (2, 64)
            assert torch.isfinite(scored["fused"]).all()
            if not with_aux:
                # An absent auxiliary cannot propose a test-time mask.
                z = model.encode(x)
                expected = model.predictor(z, None)
                assert torch.allclose(scored["fused"],
                                      (expected - z).abs().mean(dim=1))
                z_inj = model.encode(injected)
                assert torch.allclose(proposed["predicted"]["L0"],
                                      model.predictor(z_inj, None))
            print(name, "OK")


if __name__ == "__main__":
    main()
