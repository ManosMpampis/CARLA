"""Build the visual interpretation, per-machine notes, and offline gallery."""
import base64
import html
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent

NOTES={
'1-1':'The central faults have clear peaks. Normal shoulders and background excursions also exceed calibration; raising the threshold helps, while small late events remain easy to overlook.',
'1-2':'Some events are distinct and others weak. Window calibration is already close to its oracle; a different window threshold alone cannot produce a large gain.',
'1-3':'The largest late peak is a real fault, but its calibrated alarm lasts much longer than that event. Earlier events are not consistently separated from background.',
'1-4':'The late peak is genuine, yet the alarm extends far beyond it and several earlier faults are weak. Window F1 is already near its threshold ceiling.',
'1-5':'Short labeled events produce sharp high peaks. A higher threshold removes substantial background activity, but the remaining localization is not perfect.',
'1-6':'The long late fault is strongly separated after raising the timeline threshold. However, the point-F1 oracle then hits only 4 of 29 events, sacrificing many short faults.',
'1-7':'Many positive windows lie below calibration, so a lower window threshold helps. The timeline still has substantial score overlap and only moderate threshold potential.',
'1-8':'Visible peaks do not cover many of the labeled events. Window calibration is nearly optimal already; better score separation is needed for a major gain.',
'2-1':'Some peaks coincide with faults, but similar excursions occur elsewhere and event interiors can be quiet. Both calibrated scores are close to their oracle ceilings.',
'2-2':'High onset responses are visible, yet many labeled interiors have lower scores. Timeline threshold tuning barely improves F1; window averaging has more headroom.',
'2-3':'A high peak is real, but the peak-containing labeled event lasts only 6 timesteps while the calibrated alarm lasts 110. Event hits can look good while duration precision is poor.',
'2-4':'One dominant long event is reconstructed well; most other events are missed at calibration. Lowering the window threshold helps, but the point-F1 optimum focuses on long events.',
'2-5':'Several positive windows sit just below calibration. Lowering the window threshold improves F1 substantially, whereas precise timestep localization remains harder.',
'2-6':'Visible peaks contain real anomalies. Calibrated alarms have long tails; a higher timeline threshold narrows them, while the window result is already moderately useful.',
'2-7':'The dominant event is strongly detectable, but only 3 of 20 events are hit at calibration and 2 at the point-F1 oracle. High point F1 is not broad event coverage.',
'2-8':'A particularly strong calibration-failure case: almost perfect ranking and a clear real peak, but a post-event score plateau remains above the calibrated threshold.',
'2-9':'Several central faults are strongly visible. A higher threshold suppresses surrounding normal responses; the detector retains meaningful event and window information.',
'3-1':'Real early/late faults are visible, but the calibrated threshold also admits many normal excursions. Threshold repair helps; the window ceiling remains moderate.',
'3-10':'Major faults are visible, especially the late one. Periodic window-score excursions also occur outside positive windows; timestep and window usefulness differ sharply.',
'3-11':'The rare, very large fault responses support strong window detection potential. Calibration floods normal regions, and the timestep responses are much broader than short fault labels.',
'3-2':'The highest score is normal and calibration raises no alarms. Lower thresholds recover some positives at a large false-positive cost; precise localization remains weak.',
'3-3':'The highest score is normal and labeled event scores overlap background. Oracle window F1 is only modestly above always-alert F1; precise timestep discrimination is very weak.',
'3-4':'A clear first-event peak is followed by a long elevated normal-score plateau. A higher threshold detects the major fault well, but its point-F1 oracle hits only 2 of 8 events.',
'3-5':'A real peak is followed by a long calibrated alarm outside most labels. Raising the threshold helps point F1, but misses many smaller events.',
'3-6':'One of the clearest already-useful cases: multiple fault responses, good current window F1, and all 11 events hit at calibration. Some threshold headroom remains.',
'3-7':'A real onset peak is followed by a persistent elevated background. Better calibration helps the timeline, but window F1 has a low ceiling; Query genuinely performs better here.',
'3-8':'The normal-score background rises and many peaks occur. One global threshold cannot cleanly separate all events from that changing background.',
'3-9':'Real fault peaks coexist with many elevated normal responses, especially near the end. Calibration is too permissive; threshold tuning helps but does not make the window task easy.'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def pic(name,alt):return f'![{alt}]({OUT/name})'
def link(name,label=None):return f'[{label or name}]({OUT/name})'


def peaks(d,e):
    rows=[]
    for _,row in d.iterrows():
        z=np.load(OUT/'data'/row.machine.replace('.txt','.npz'));s=z['replay_scores'];y=z['labels'];t=int(row.peak_timestep)
        ev=e[(e.machine==row.machine)&(e.start<=t)&(e.end>t)]
        rec=dict(machine=row.machine,peak=t,peak_is_anomaly=bool(row.peak_is_anomaly),
                 peak_event_duration=int(ev.iloc[0].duration) if len(ev) else None)
        for name,threshold in [('cal',row.timeline_threshold),('oracle',row.timeline_oracle_threshold)]:
            pred=s>threshold
            if not pred[t]:
                rec.update({name+'_alarm_start':None,name+'_alarm_end':None,name+'_alarm_duration':0,name+'_alarm_precision':None})
                continue
            left=t;right=t+1
            while left>0 and pred[left-1]:left-=1
            while right<len(s) and pred[right]:right+=1
            rec.update({name+'_alarm_start':left,name+'_alarm_end':right,name+'_alarm_duration':right-left,
                        name+'_alarm_precision':float(y[left:right].mean()),name+'_alarm_right_censored':right==len(s)})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT/'peak_localization.csv',index=False)


def main():
    d=pd.read_csv(OUT/'machine_summary.csv');e=pd.read_csv(OUT/'events.csv');peaks(d,e)
    means=d.mean(numeric_only=True)
    machine_rows=[]
    for _,r in d.iterrows():
        name=r.machine.removesuffix('.txt');key=name.removeprefix('machine-')
        machine_rows.append([link(f'machines/{name}.png',key),f'{r.window_cal_f1:.3f} → {r.window_oracle_f1:.3f}',
            f'{r.timeline_cal_f1:.3f} → {r.timeline_oracle_f1:.3f}',
            f'{int(r.events_calibrated_hit)}/{int(r.timeline_events)}',f'{r.timeline_cal_fpr:.1%}',NOTES[key]])
    alltable=table(['Machine / full image','Window F1 calibrated → oracle','Timeline F1 calibrated → oracle','Covered events hit at calibration','Normal timestep FPR','Visual interpretation'],machine_rows)
    report=f'''# Are the frequency_aux observed-features scores useful?

30 October 2026 · SMD · all 28 machines · `lewm_encoder/frequency_aux/cross_attention/observed_features` · selected validation-loss weights. No retraining or fresh model inference.

**Yes: your observation is supported by the actual traces. This head contains useful anomaly information on many machines. Its low average calibrated F1 does not establish that cross attention has failed.** The largest final replay score lies inside a labeled anomaly on **26/28 machines**. Among the highest-scoring 1% of covered timesteps, mean per-machine precision is **58.7%**, compared with mean anomaly prevalence **4.29%**. That is a substantial concentration of fault evidence.

The qualified conclusion is that **some machines mainly have a calibration problem; others also have an alarm-duration/localization problem; a smaller set has genuinely weak score separation**. One threshold change cannot solve all three. Ten machines have window-oracle F1 ≥0.70, while four remain ≤0.50 even with a test-chosen window threshold. The pointwise ceiling is different: six machines reach ≥0.70 and seven remain ≤0.35.

I loaded **every `decision_trace.json`**, reconstructed the final timeline and the original window contributions, and saved a graph for every machine. Browser policy blocks local `file:` pages, so these are **renderings of the actual JSON graph values, not screenshots of the HTML UI**. Three original HTML embedded payloads were independently checked against their JSON. The {link('GALLERY.html','offline image gallery')} contains all 28 images; the machine table below links individual figures.

## 1. What the graph and window metric measure

| View | Score | Label | Threshold |
| --- | --- | --- | --- |
| HTML/full-series replay | Mean of all overlapping contributions to a timestep | Actual anomaly label at that timestep | Timestep clean-calibrated threshold |
| Window evaluation | Mean error across one 512-timestep scored target | Positive if **any** anomaly occurs anywhere in those 512 timesteps | Separate window clean-calibrated threshold |

A single obvious fault peak may be exactly where you expect and still have a poor score for different reasons:

- Its response extends outside the labeled event, producing false-positive duration.
- It catches the onset but stays low during the rest of the fault, producing missed duration.
- Mean pooling dilutes a short peak across 512 positions, while the any-positive window label covers many windows around that short event.
- It detects one visually dominant event but misses many smaller ones.
- Normal score levels change across the series, so one global threshold cannot separate every regime.

The first 512 timesteps provide context and receive no scores. Across the 28 recordings, five labeled events lie entirely in that startup region; **all event denominators here exclude those five unscored events**, while the pictures still show their truth bands. There are 322 covered events and 327 total events. Timestep metrics likewise use covered positions. This coverage restriction is shared by the matched heads.

These are measurable effects here, not reasons to disregard the visual evidence. Also, a better threshold changes F1 but **does not change VUS** for an unchanged score vector: VUS already integrates threshold-based ranking behavior. A window VUS value must be interpreted using the window score/label definition, not inferred from the different timestep curve.

In the pictures, blue is the actual score, green dashed is calibration, and orange dotted is the separately optimized test oracle. Scores are divided by their own calibrated threshold, so the current threshold is always **1**. The vertical transform is the HTML's nonnegative `log1p(score/threshold)` transform. The final curve gets a fitted vertical range instead of the HTML's full historical range; the pixels and UI are not identical. No smoothing, clipping, or label-based score transformation is applied. Purple denotes actual timestep faults in the top panel and **positive-window center ranges** in the middle panel. Window centers are display positions, not causal alarm times.

## 2. Your strongest example: useful signal hidden by calibration

### Machine-2-8

The highest response aligns with the only labeled event, and the separation is excellent:

| Diagnostic | Calibrated | Oracle on the same saved scores |
| --- | ---: | ---: |
| Window F1 | 0.0919 | **0.9771** |
| Timestep F1 | 0.0296 | **0.9369** |
| Normal timestep false-positive rate | 45.8% | 0.07% |

The event lasts **161 timesteps**. At calibration the peak-containing continuous alarm spans **6,207 recorded timesteps and continues to the recording's end**, including the elevated post-event baseline. At the point-F1 oracle it lasts **172 timesteps**, with precision **90.7%**. Ordinary timestep AP is **0.9899**, ROC **0.9999**; window VUS-PR/ROC are **0.9411/0.9820**. This is direct evidence that the representation can distinguish the event and that the supplied threshold makes it look much worse than its ranking warrants.

{pic('machines/machine-2-8.png','Machine-2-8: excellent signal, poor calibrated threshold')}

The oracle is a retrospective ceiling chosen with test labels, not a threshold the existing clean-tail method knows how to find. Nevertheless, this case strongly supports keeping the architecture and investigating calibration.

### Machine-3-11

Window F1 improves **0.0947 → 0.9174**, with window VUS-PR/ROC **0.9335/0.9832**. All three labeled events are hit at calibration, but **61.8%** of normal timesteps also alert. There is clear window-level potential. The best timestep F1 is only **0.4961**, so improving the threshold does not completely solve duration localization. At the final maximum, a 6-timestep labeled event sits inside a 146-timestep alarm even at the point-F1 oracle; nearby labeled events can also contribute to the alarm's precision.

### Machine-3-6

This is useful even with the current threshold: window F1 **0.7759**, timestep F1 **0.6153**, and **11/11 events hit**, with normal timestep FPR **4.6%**. The corresponding oracle ceilings are **0.8705 / 0.7365**. It is a good reference case for inspecting what this head learns successfully.

## 3. Why a convincing peak can still yield low F1

### The peak is right, but its alarm is too broad

On **machine-2-3**, the maximum lies in a **6-timestep** labeled event. The calibrated alarm around it spans **110 timesteps**; the oracle alarm still spans **94**. At calibration the head hits **9/10 events**, yet timestep F1 is only **0.1069**. Detecting the event and localizing its duration are different successes. Window F1 is much higher, **0.5248**, but already close to its **0.5319** oracle ceiling.

On **machine-1-4**, the maximum is also a true anomaly. Its containing labeled event lasts **46 timesteps**, while the calibrated peak alarm lasts **773**. Earlier faults are not consistently separated either. Window F1 is **0.5446 → 0.5458**: here calibration is already near-optimal for the existing window score, so a different window threshold cannot deliver a dramatic gain.

Coarse encoded features, receptive fields and overlapping reconstruction can plausibly broaden responses. This audit establishes the width mismatch but does not isolate which mechanism caused it. Very long post-event plateaus can exceed the whole 1,024-timestep input length; overlap alone cannot explain every such plateau in a stateless window scorer. Those cases need inspection of the nominal telemetry regime and model error after the event.

### One large fault can dominate point F1 while many events are missed

On **machine-1-6**, timestep F1 improves **0.3176 → 0.8840**. But the point-F1 oracle hits only **4/29 events**, versus **28/29** at calibration: it removes many small-event alarms while concentrating on the long late fault. On **machine-2-7**, timestep oracle F1 is **0.8224**, yet only **2/20 events** are hit. These can be useful detectors of dominant faults, but those F1 values do not mean broad fault coverage.

The event-hit counts here are the simple fraction of covered labeled intervals containing at least one positive prediction. They are a diagnostic, **not** point-adjusted F1 and not a full event matching metric. A high event hit rate can still come with many false alerts.

### Sometimes the scores themselves are weak

On **machine-3-2** and **machine-3-3**, the maximum score is outside every labeled fault. Timestep oracle F1 is **0.1349 / 0.0744**, and recall at ≤1% ordinary timestep FPR is only **3.0% / 3.0%**. These are not rescued by choosing a better global threshold.

Machine-3-3's window oracle F1 is **0.7457**, but an always-alert window classifier already obtains **0.7071** because 54.7% of scored windows contain some anomaly. That window F1 should not overrule the weak timestep separation visible in the graph. It is a different, high-prevalence labeling task.

{pic('machines/machine-3-3.png','Machine-3-3: genuinely weak timestep separation and a high-prevalence window task')}

## 4. What happens across all machines

All means below give each machine equal weight. Timeline statistics come from the JSON final states; window statistics use the official NPZ window scores.

| Quantity | Mean across 28 machines |
| --- | ---: |
| Window calibrated / oracle F1 | **0.4046 / 0.6489** |
| Timestep calibrated / oracle F1 | **0.2308 / 0.5022** |
| Ordinary timestep AP / ROC | **0.4632 / 0.8811** |
| Event hit fraction, calibrated / timestep-oracle threshold | **70.3% / 56.7%** |
| Normal timestep FPR, calibrated / timestep-oracle threshold | **21.2% / 5.18%** |
| Timestep recall at ≤1% ordinary FPR | **40.4%** |

This is useful ranking information, but the default threshold has substantial false-alarm burden. The oracle raises ordinary point F1 on average while reducing equal-weight event coverage. A lower-FP operational threshold and an event-sensitive objective may therefore favor different settings. The 1% FPR diagnostic uses test labels to characterize the curve, not a deployable threshold selection rule.

### Is observed-features actually worse than other cross-attention heads?

The matched comparison uses identical machines, input length 1,024, target length 512, stride 10, labels and scored support. These are the right controls for judging this architectural head.

| Variant | Window calibrated F1 | Window oracle F1 | Window VUS-PR | Window VUS-ROC |
| --- | ---: | ---: | ---: | ---: |
| frequency_aux observed_features | 0.4046 | **0.6489** | 0.5655 | 0.7024 |
| frequency_aux observed_query | 0.4150 | 0.6404 | 0.5692 | 0.7059 |
| frequency_aux observed_input | 0.3450 | 0.5852 | 0.4980 | 0.6512 |
| frequency_aux_no_film observed_features | 0.4060 | 0.6564 | 0.5647 | 0.6984 |

**Observed-features is competitive with Query and no-FiLM Features, and clearly stronger than observed Input on most machines.** It beats observed Input on calibrated window F1 on **20/28** machines and on oracle F1 on **24/28**. Against Query, calibrated wins split **14/14**, while oracle F1 gives Features **17 wins, 10 losses, 1 tie**. Its matched NPZ timestep AP/ROC are **0.4632/0.8811**, versus Query's **0.4532/0.8666**. These are descriptive comparisons at seed 4, not proof of a statistically established global winner.

There are genuine counterexamples: Query is better on machine-3-7, and no-FiLM Features is better on machine-1-2. The {link('matched_heads.md','matched-head comparison')} details those cases. The earlier full-span-versus-target-half headline averages cannot establish that cross attention is architecturally inferior, because those methods have different scored regions and window labels.

## 5. Machine-by-machine visual judgment

Every row links a full-resolution picture: timeline, window means, and both threshold/F1 curves. “Oracle” means the best ordinary F1 on that particular saved test score vector; the timeline and window oracles are separate.

{alltable}

## 6. All-machine contact sheets

These cover every machine without selecting only attractive examples. The individual figures retain more detail than these compact pages.

{pic('contact_1.png','Machines 1-1 through 1-7')}

{pic('contact_2.png','Machines 1-8 through 2-6')}

{pic('contact_3.png','Machines 2-7 through 3-2, including 3-10 and 3-11')}

{pic('contact_4.png','Machines 3-3 through 3-9')}

## 7. What I would change in the investigation

1. **Keep this feature head in the shortlist.** The matched comparisons and real fault peaks justify it. Do not discard it on the strength of one macro calibrated F1.
2. **Work first on the clear calibration cases:** machine-2-8, machine-3-11, machine-3-4, and machine-1-6. Keep weights fixed and test threshold transfer on disjoint nominal blocks. The existing nominal tail and extreme quantile are a thin basis for a stable threshold; simply retuning on test labels would not establish a usable calibration method.
3. **Define whether the intended alarm is event onset or full fault duration.** If onset is the operational goal, evaluate event coverage, grouped false alarms and detection delay honestly. Do not turn peak hits into point-adjusted F1 and call localization solved. Final replay scores can use later overlapping windows, so this audit does not establish causal online latency.
4. **Treat duration errors separately:** inspect the raw dense score map, target/context influence and post-event nominal telemetry. Test whether a different score reduction preserves short peaks without extending false alarms, holding weights and window labels fixed.
5. **Use weak-separation controls:** machine-3-2, machine-3-3, machine-3-7, and machine-3-8. If the best threshold is still insufficient, improve the learned score or head there rather than continuing to search quantiles.

The evidence supports your view that many values are useful. It also explains why the evaluation is not simply wrong: **the visible peak, the usable threshold, event coverage, and alarm duration are four distinct properties**. This architecture demonstrates the first strongly on many machines; the other properties vary substantially.

## 8. Data checks and reproducibility

All 28 JSON traces cover their full saved evaluation, without truncation. Their coverage counts, scored support and truth intervals match the NPZ arrays exactly. The original HTML/JSON payload equality was verified for three samples. Window calibrated and oracle F1 were reproduced from the official score vectors.

Replay scores are close to, but not bitwise identical to, batched NPZ scores. Across all 28 machines there are **8 changed timestep decisions in total**, at the recorded calibrated thresholds; the largest calibrated timestep F1 difference is **0.000071**. One machine has one recovered-window decision difference. The trace is produced one input at a time, while NPZ scoring uses batches; that is a plausible numerical explanation, not a proven cause. These residuals do not explain the major performance gaps. Both are retained in {link('machine_summary.csv')}.

Sources and details: {link('trace_semantics.md','trace source audit')}, {link('events.csv','per-event evidence')}, {link('peak_localization.csv','peak alarm durations')}, {link('matched_heads.csv','matched score comparisons')}, {link('trace_sample_verification.json','HTML/JSON sample verification')}. The compact `data/` archives contain the extracted final graph arrays; no model artifacts were edited.

```bash
./venv/bin/python reports/smd_cross_attention_visual_audit/analyze_traces.py
./venv/bin/python reports/smd_cross_attention_visual_audit/compare_matched_heads.py
./venv/bin/python reports/smd_cross_attention_visual_audit/verify_trace_samples.py
./venv/bin/python reports/smd_cross_attention_visual_audit/write_report.py
```
'''
    (OUT/'REPORT.md').write_text(report)
    cards=[]
    for _,r in d.iterrows():
        name=r.machine.removesuffix('.txt');key=name.removeprefix('machine-')
        img=base64.b64encode((OUT/'machines'/f'{name}.png').read_bytes()).decode()
        cards.append(f'<section id="{name}"><h2>{name}</h2><p>{html.escape(NOTES[key])}</p><img alt="{name} score graphs" src="data:image/png;base64,{img}"></section>')
    nav=' · '.join(f'<a href="#machine-{k}">{k}</a>' for k in NOTES)
    gallery=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SMD observed-features score gallery</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#23333c;background:#f6f8fa;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}h1,h2{{line-height:1.2}}nav,section{{background:white;padding:22px;margin:20px 0;border:1px solid #dce3e7;border-radius:12px}}nav{{position:sticky;top:0;z-index:2;font-size:14px}}a{{color:#176f83}}img{{width:100%;height:auto}}section{{scroll-margin-top:115px}}</style>
<main><h1>Frequency_aux observed-features: every SMD machine</h1><p>30 October 2026 · Exact final replay data reconstructed from JSON, with official window-score comparison. These are data renderings, not browser screenshots. Purple bands: true anomalies in top panel, positive-window ranges in middle. Green dashed: calibrated threshold. Orange dotted: retrospective oracle. Score units are divided by the appropriate calibrated threshold; vertical scale is log1p.</p><p>Initial 512 timesteps are context-only. Event totals exclude five wholly unscored startup events (322 covered, 327 total). All thresholds remain unchanged in experiment artifacts. Oracle choices use test labels and demonstrate potential; they are not new calibrated thresholds.</p><nav>{nav}</nav>{''.join(cards)}</main></html>'''
    (OUT/'GALLERY.html').write_text(gallery)
    print(f'Wrote visual report ({len(report.split())} words), 28-machine offline gallery, peak-localization audit.')


if __name__=='__main__':main()
