"""Independent model/loss acceptance for the approved conditional LeWM spec.

Run with ./venv/bin/python tests/verify_conditional_models.py [--output report.json].
Expected values come from the agreed equations, not snapshots of model output.
"""

import argparse
import copy
import math

from conditional_support import close, preset, run_cases, tiny_config

import torch
from torch import nn
from torch.nn import functional as F

from losses.lewm import LeWMLoss
from models import get_model
from models.ae_baseline import reconstruction_objective
from utils.common_config import get_criterion
from utils.reconstruction_scores import reconstruction_score_map


def default_contract():
    details = {}
    for name in ("lewm", "guided_forecast"):
        config = preset(name)
        criterion = get_criterion(config)
        assert criterion.loss2_kind == "kl", "shipped LeWM preset must default to KL"
        assert config["score_mode"] == "l1", "score choice is independent, default L1"
        assert criterion.w_recon == 0, "raw reconstruction is opt-in"
        model = get_model(tiny_config(name), input_channels=3)
        assert model.head is None, "default latent predictor has no raw head"
        assert model.total_stride == 1, "main experiment uses stride 1"
        if name == "lewm":
            assert criterion.w_aux == 1 and criterion.mask_loss_kind == "bce"
        else:
            assert criterion.w_aux == 0
        assert criterion.w_pred == 1
        assert criterion.lambda_sigreg == criterion.lambda_sigreg_tgt == 0.1
        details[name] = {"latent_loss":criterion.loss2_kind, "score":config["score_mode"]}
    return details


def objective_equations():
    prediction = torch.tensor([[[0., 2.], [0., 4.]]], requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    p = [math.exp(2)/(math.exp(2)+math.exp(4)), math.exp(4)/(math.exp(2)+math.exp(4))]
    kl_second = 0.5 * sum((a-0.5)*math.log(a/0.5) for a in p)
    expected = {"l1":1.5, "mse":5., "l2":math.sqrt(10)/2, "kl":kl_second/2}
    values = {}
    for kind in expected:
        criterion = LeWMLoss(loss2_kind=kind, w_aux=0, lambda_sigreg=0, lambda_sigreg_tgt=0)
        result = criterion({"predicted": prediction, "targets": target})
        close(result["loss"], expected[kind])
        values[kind] = float(result["loss"].detach())
        gradients = torch.autograd.grad(result["loss"], (prediction, target), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gradients)
        assert all(g.abs().sum() > 0 for g in gradients), "shared target must receive gradients"
    exact = torch.zeros(2, 4, 3, requires_grad=True)
    zero_loss = LeWMLoss(loss2_kind="l2", w_aux=0, lambda_sigreg=0, lambda_sigreg_tgt=0)(
        {"predicted": exact, "targets": torch.zeros_like(exact)})["loss"]
    close(zero_loss, 0.)
    zero_loss.backward()
    assert torch.isfinite(exact.grad).all(), "zero-error RMS gradient must be finite"
    close(exact.grad, torch.zeros_like(exact))
    return values


def independent_score_equations():
    prediction = torch.tensor([[[0., 2.], [0., 4.]]])
    target = torch.zeros_like(prediction)
    p = [math.exp(2)/(math.exp(2)+math.exp(4)), math.exp(4)/(math.exp(2)+math.exp(4))]
    kl = 0.5 * sum((a-0.5)*math.log(a/0.5) for a in p)
    expected = {"l1":[[0.,3.]], "mse":[[0.,10.]], "l2":[[0.,math.sqrt(10)]], "kl":[[0.,kl]]}
    for kind, value in expected.items():
        scores, _ = reconstruction_score_map(prediction, target, kind)
        close(scores, value)
        assert scores.dtype == torch.float32
    shifted = target + 7
    close(reconstruction_score_map(shifted, target, "kl")[0], [[0.,0.]])
    close(reconstruction_score_map(shifted, target, "l1")[0], [[7.,7.]])
    return expected


def spectral_coordinates():
    model = get_model(tiny_config(), input_channels=3).eval()
    predictor = model.predictor
    seen = {}
    handles = [
        predictor.stem_conv.register_forward_hook(lambda m, a, out: seen.update(stem=out.detach())),
        predictor.in_proj.register_forward_pre_hook(lambda m, a: seen.update(packed=a[0].detach())),
        predictor.out_proj.register_forward_hook(lambda m, a, out: seen.update(spectral=out.detach())),
        predictor.to_latent.register_forward_pre_hook(lambda m, a: seen.update(reconstructed=a[0].detach())),
    ]
    shapes = []
    try:
        for batch, length in ((1,32),(2,35),(3,48)):
            z = torch.arange(batch*6*length, dtype=torch.float32).reshape(batch,6,length)/50
            auxiliary = torch.randn(batch,4,length)
            output = predictor(z, auxiliary)
            assert output.shape == z.shape
            n_fft, hop, win = predictor._resolve_stft(length)
            window = torch.hann_window(win)
            spec = torch.stft(seen["stem"].reshape(batch*4,length).float(), n_fft=n_fft,
                              hop_length=hop, win_length=win, window=window, center=True,
                              return_complex=True)
            frequencies, frames = spec.shape[-2:]
            expected = torch.cat([spec.real.reshape(batch,4,frequencies,frames).transpose(-1,-2),
                                  spec.imag.reshape(batch,4,frequencies,frames).transpose(-1,-2)],dim=1)
            close(seen["packed"],expected)
            real, imag = seen["spectral"].chunk(2,dim=1)
            complex_output = torch.complex(real,imag).transpose(-1,-2).reshape(batch*4,frequencies,frames)
            rec = torch.istft(complex_output,n_fft=n_fft,hop_length=hop,win_length=win,
                              window=window,center=True,length=length).reshape(batch,4,length)
            close(seen["reconstructed"],rec)
            # Independent unchanged-transform roundtrip checks all coordinates.
            real, imag = seen["packed"].chunk(2,dim=1)
            identity = torch.complex(real,imag).transpose(-1,-2).reshape(batch*4,frequencies,frames)
            restored = torch.istft(identity,n_fft=n_fft,hop_length=hop,win_length=win,
                                   window=window,center=True,length=length).reshape(batch,4,length)
            close(restored,seen["stem"],atol=3e-6)
            shapes.append({"input":list(z.shape),"packed":list(expected.shape)})
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def predicted_mask_and_gradients():
    model = get_model(tiny_config(),input_channels=3).train()
    x = torch.randn(3,3,48)
    known = torch.zeros(3,48,dtype=torch.bool)
    known[:,8:20] = True
    corrupted = x.clone()
    corrupted[:,:,8:20] += 4
    outputs = model(x,mask={"input":known,"X_inj":corrupted})
    changed_labels = model(x,mask={"input":~known,"X_inj":corrupted})
    close(outputs["action"].mask_proposal,outputs["mask_logits"].sigmoid())
    close(outputs["predicted"],changed_labels["predicted"])
    assert not torch.equal(outputs["mask_target"],changed_labels["mask_target"])
    assert "recon" not in outputs
    target = outputs["targets"]
    target.retain_grad()
    criterion = get_criterion(tiny_config())
    losses = criterion(outputs)
    expected = (losses["pred_loss"]+losses["aux_loss"]+
                0.1*losses["sigreg"]+0.1*losses["sigreg_tgt"])
    close(losses["loss"],expected)
    close(losses["aux_loss"],F.binary_cross_entropy_with_logits(outputs["mask_logits"],known.float()))
    losses["loss"].backward()
    assert target.grad is not None and target.grad.abs().sum()>0
    for role in (model.encoder,model.action.aux,model.predictor):
        grads=[p.grad for p in role.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    return {name:float(value.detach()) for name,value in losses.items()}


def unrestricted_latent():
    model=get_model(tiny_config(),input_channels=3).eval()
    with torch.no_grad():
        for layer in model.encoder.modules():
            if isinstance(layer,nn.Conv1d):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.fill_(-2.)
        encoded=model.encode(torch.randn(2,3,48))
    assert encoded.max() < -1, "latent output must permit values below GELU branch lower bound"
    return {"latent_min":float(encoded.min()),"latent_max":float(encoded.max())}


def guided_roles_and_shapes():
    config=tiny_config("guided_forecast")
    model=get_model(config,input_channels=3).eval()
    assert any(isinstance(m,nn.ConvTranspose1d) for m in model.predictor.modules())
    assert not any(isinstance(m,nn.Linear) for m in model.modules())
    seen=[]
    handle=model.predictor.attention.register_forward_pre_hook(
        lambda m,a,kw:seen.append({k:v.detach().clone() for k,v in kw.items() if torch.is_tensor(v)}),
        with_kwargs=True)
    details=[]
    try:
        for length in (48,63,90):
            x=torch.randn(2,3,length)
            out=model(x)
            start,end=out["target_range"]
            qkv=seen[-1]
            assert start>=1 and end==length
            assert abs(start-2*length/3)<=1
            assert qkv["q"].shape[-1]==length-start
            assert qkv["k"].shape[-1]==qkv["v"].shape[-1]==start
            assert out["predicted"].shape==out["targets"].shape==(2,6,length-start)
            assert "recon" not in out
            before={k:v.clone() for k,v in qkv.items()}
            future_changed=x.clone();future_changed[:,:,start:]+=2
            model(future_changed)
            close(seen[-1]["k"],before["k"]);close(seen[-1]["v"],before["v"])
            assert not torch.allclose(seen[-1]["q"],before["q"])
            history_changed=x.clone();history_changed[:,:,:start]+=2
            model(history_changed)
            close(seen[-1]["q"],before["q"])
            assert not torch.allclose(seen[-1]["k"],before["k"])
            assert not torch.allclose(seen[-1]["v"],before["v"])
            details.append({"input":list(x.shape),"history_tokens":start,"future_tokens":length-start})
    finally:
        handle.remove()
    return details


def score_interpolation():
    details=[]
    for name in ("lewm","guided_forecast"):
        config=tiny_config(name,stride=2)
        model=get_model(config,input_channels=3).eval()
        x=torch.randn(2,3,66)
        for mode in ("l1","kl","mse","l2"):
            model.score_mode=mode
            out=model(x,corrupt=False)
            expected,_=reconstruction_score_map(out["predicted"],out["targets"],mode)
            start,end=out["target_range"]
            expected=F.interpolate(expected.unsqueeze(1),size=end-start,mode="linear",align_corners=False).squeeze(1)
            actual=model.score(x)["fused"]
            close(actual,expected)
            assert actual.shape==(2,end-start) and actual.dtype==torch.float32
        details.append({"model":name,"span":[start,end],"stride":model.total_stride})
    return details


def optional_raw_and_weighted_total():
    config=tiny_config()
    config["lewm_modules"]["head"]={"name":"mirrored","norm":"none","dropout":0.}
    model=get_model(config,input_channels=3)
    outputs=model(torch.randn(3,3,48))
    assert outputs["recon"].shape==(3,3,48)
    criterion=LeWMLoss(w_pred=2.,w_aux=3.,w_recon=.5,lambda_sigreg=.2,lambda_sigreg_tgt=.4,
                       loss2_kind="kl",mask_loss_kind="bce",require_mask=True,
                       sigreg_kwargs={"num_slices":4,"freq_nodes":5})
    terms=criterion(outputs)
    expected=2*terms["pred_loss"]+3*terms["aux_loss"]+.5*terms["recon_loss"]+.2*terms["sigreg"]+.4*terms["sigreg_tgt"]
    close(terms["loss"],expected)
    terms["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    zero=LeWMLoss(w_pred=0,w_aux=0,w_recon=0,lambda_sigreg=0,lambda_sigreg_tgt=0)
    # Disabled losses do not require target/context/auxiliary/raw operands.
    close(zero({"predicted": torch.zeros(2,4,8)})["loss"],0.)
    try:
        LeWMLoss(w_aux=0,lambda_sigreg=0,lambda_sigreg_tgt=0)(
            {"predicted": torch.zeros(2,4,1),"targets": torch.zeros(2,4,8)})
    except ValueError:
        pass
    else:
        raise AssertionError("shape broadcasting must be rejected")
    return {key:float(value.detach()) for key,value in terms.items()}


def fixed_batch_optimization_and_conditioning():
    details={}
    for name in ("lewm","guided_forecast"):
        config=tiny_config(name)
        model=get_model(config,input_channels=3)
        criterion=get_criterion(config)
        optimizer=torch.optim.Adam(model.parameters(),lr=.004)
        t=torch.linspace(0,5,48)
        x=torch.stack([torch.stack([torch.sin(t+i+j/2) for j in range(3)]) for i in range(4)])
        known=torch.zeros(4,48,dtype=torch.bool);known[:,12:25]=True
        injected=x.clone();injected[:,:,12:25]+=2
        mask={"input":known,"X_inj":injected} if name=="lewm" else None
        losses=[]
        for step in range(45):
            model.train();optimizer.zero_grad()
            # Hold regularizer sketches fixed only for the overfit diagnostic.
            torch.manual_seed(700)
            output=model(x,mask=mask)
            total=criterion(output)["loss"]
            total.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step();losses.append(float(total.detach()))
        assert min(losses[-5:])<losses[0]*.8,"fixed-batch objective must demonstrably decrease"
        model.eval()
        out=model(x,mask=mask)
        current=out["context"]
        action=out["action"]
        baseline=model.predictor.predict(current,action,target_tokens=out["targets"].size(-1))
        changed=copy.copy(action)
        if name=="lewm":
            changed.film_feat=action.film_feat.roll(1,0)
        else:
            changed.query=action.query.roll(1,0)
        guide_shuffled=model.predictor.predict(current,changed,target_tokens=baseline.size(-1))
        history_shuffled=model.predictor.predict(current.roll(1,0),action,target_tokens=baseline.size(-1))
        guide_effect=float((baseline-guide_shuffled).abs().mean().detach())
        history_effect=float((baseline-history_shuffled).abs().mean().detach())
        assert guide_effect>1e-7,"trained guide must affect predictions with current latent fixed"
        if name!="lewm":assert history_effect>1e-7
        # For masked reconstruction the predictor consumes action.token_input;
        # alter that actual current-state operand to test its influence.
        if name=="lewm":
            changed=copy.copy(action);changed.token_input=action.token_input.roll(1,0)
            history_shuffled=model.predictor.predict(current,changed,target_tokens=baseline.size(-1))
            history_effect=float((baseline-history_shuffled).abs().mean().detach())
            assert history_effect>1e-7
        details[name]={"initial_loss":losses[0],"final_loss":losses[-1],"steps":45,
                       "guide_shuffle_mean_difference":guide_effect,
                       "history_shuffle_mean_difference":history_effect}
    return details


def cuda_amp_frequency():
    if not torch.cuda.is_available():
        return {"available":False,"limitation":"CUDA AMP not exercised"}
    config=tiny_config()
    model=get_model(config,input_channels=3).cuda().train()
    criterion=get_criterion(config).cuda()
    x=torch.randn(3,3,48,device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        outputs=model(x)
        terms=criterion(outputs)
    terms["loss"].backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert torch.isfinite(terms["loss"])
    score=model.score(x)["fused"]
    assert score.dtype==torch.float32 and torch.isfinite(score).all()
    return {"available":True,"device":torch.cuda.get_device_name(),
            "loss":float(terms["loss"].detach()),"score_dtype":str(score.dtype)}


def baseline_objectives_and_widths():
    target=torch.zeros(2,3,7)
    outputs={"recon":torch.ones_like(target),"mu":torch.full((2,5,7),2.),
             "logvar":torch.full((2,5,7),math.log(4.))}
    result=reconstruction_objective(outputs,target,.3)
    expected_kl=.5*(4+4-1-math.log(4))
    close(result["mse"],1.);close(result["kl"],expected_kl)
    close(result["loss"],1+.3*expected_kl)
    output_shapes=[]
    for name in ("ae","vae","lewm","guided_forecast"):
        for channels in (1,5):
            config=tiny_config(name,channels=channels)
            model=get_model(config,input_channels=channels)
            x=torch.randn(2,channels,48)
            if name=="vae":
                model.train()
                first=model(x);second=model(x)
                assert first["mu"].shape==(2,6,48)
                assert not torch.equal(first["recon"],second["recon"])
                model.eval()
                close(model.score(x)["fused"],model.score(x)["fused"],atol=0,rtol=0)
            model.eval()
            for length in (48,65):
                value=model(torch.randn(2,channels,length))
                if name in ("ae","vae"):assert value["recon"].shape==(2,channels,length)
                else:assert value["predicted"].shape[1]==6
                output_shapes.append({"model":name,"raw_channels":channels,"window":length})
    return {"vae_kl":expected_kl,"cases":output_shapes}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args=parser.parse_args()
    cases=[("shipped defaults",default_contract),("literal objective equations",objective_equations),
           ("independent score equations",independent_score_equations),
           ("spectral coordinate roundtrip",spectral_coordinates),
           ("predicted mask and gradients",predicted_mask_and_gradients),
           ("unrestricted latent",unrestricted_latent),("future Q historical KV",guided_roles_and_shapes),
           ("scalar score interpolation",score_interpolation),
           ("optional raw weighted total",optional_raw_and_weighted_total),
           ("fixed batch optimization conditioning",fixed_batch_optimization_and_conditioning),
           ("CUDA AMP frequency",cuda_amp_frequency),("baseline objectives variable widths",baseline_objectives_and_widths)]
    results=run_cases(cases,args.output)
    raise SystemExit(any(row["status"]!="passed" for row in results))
