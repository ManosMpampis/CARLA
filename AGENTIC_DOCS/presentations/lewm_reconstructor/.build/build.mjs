import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { finalizePresentation } from '/home/manos/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root='/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA';
const workspaceDir=path.join(root,'AGENTIC_DOCS/presentations/lewm_reconstructor');
const build=path.join(workspaceDir,'.build');
const skill='/home/manos/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const p=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#11263B', muted='#526779', teal='#00867D', pale='#EFF5F5';
function text(s,content,x,y,w,h,size=24,color=navy,bold=false){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=content; sh.text.style={typeface:'Arial',fontSize:size,color,bold};
 return sh;
}
function slide(title,subtitle,num){
 const s=p.slides.add();s.background.fill='#FAFCFC';
 text(s,title,60,44,1160,68,46,navy,true);
 text(s,subtitle,60,121,1150,57,25,muted);
 text(s,num,1160,655,60,28,20,muted);
 return s;
}
const a=slide('LeWM: learning to reconstruct clean latents','Phase one trains a shared encoder on normal telemetry with synthetic corruptions.','01');
text(a,'TRAINING TASK',60,215,470,30,19,teal,true);
text(a,'Recover the clean representation',60,258,480,88,34,navy,true);
text(a,'Encode the clean and corrupted windows with the same ResNet weights. Replace selected latent positions with a learned mask token.',60,365,460,130,25);
text(a,'A time or frequency predictor reconstructs the clean latents. An optional auxiliary head detects corruption and supplies FiLM steering.',60,513,460,126,25);
text(a,'Clean target',620,215,560,32,22,muted);
text(a,'z = E(x)',620,256,560,57,42,navy,true);
text(a,'Corrupted view + mask token',620,341,560,33,22,muted);
text(a,'ẑ = P(mask(E(xᵢₙⱼ), m))',620,382,580,62,35,teal,true);
text(a,'Learning objective',620,479,560,30,22,muted);
text(a,'Latent matching + SIGReg',620,521,570,47,31,navy,true);
text(a,'Both encoder paths receive gradients.\nSIGReg discourages latent collapse.',620,580,560,65,23,muted);
a.speakerNotes.text=`Scope: the current repository's LeWM adaptation to time-series anomaly detection. This is latent reconstruction within a window, rather than a claim about next-step forecasting. x is a clean normal-training window, x_inj its synthetically corrupted version, m the injected mask, E the shared encoder, and P the predictor. Training uses the injected mask to replace predictor-input latent positions with a learned mask token. Optional auxiliary features enter P via FiLM. Both encoder passes receive gradients, with no stop-gradient or EMA teacher. The current LeWMLoss default is symmetric channel-wise KL between predicted and clean latents, plus SIGReg on corrupted-context and clean-target latents. Auxiliary variants add MSE between sigmoid(mask_logits) and the injected mask. SIGReg is not applied to the predictor output in the implementation, despite an outdated module-level loss docstring. Predictor-domain and auxiliary/FiLM choices vary by experiment. Sources: ${root}/models/lewm.py (LeWMModel.forward, TimePredictor, FreqPredictor, TimeAuxiliary); ${root}/losses/lewm.py (LeWMLoss.forward); ${root}/configs/lewm_encoder/PHASE1.md; ${root}/configs/lewm_encoder/frequency_aux/phase1.yml.`;

const b=slide('LeWM encoder reuse in the signal reconstructor','The repository names this downstream reconstruction step phase two.','02');
text(b,'ENCODER HANDOFF',60,215,470,30,19,teal,true);
text(b,'Keep the encoder.\nTrain a mirrored decoder.',60,258,500,94,34,navy,true);
text(b,'Load the phase-one encoder checkpoint. The default freezes the encoder and trains a new decoder on full normal windows.',60,376,465,116,25);
text(b,'The decoder reverses the encoder’s strides with transposed convolutions, then projects back to the sensor channels.',60,516,465,113,25);
text(b,'Raw-signal reconstruction',620,215,570,32,22,muted);
text(b,'x̂ = D(E(x))',620,256,560,57,42,teal,true);
text(b,'Example: 38 sensors, 512 timesteps',620,335,570,32,22,muted);
text(b,'Input 38 × 512     Latent 128 × 64\nOutput 38 × 512',620,380,575,77,28,navy,true);
text(b,'Training loss: mean absolute error',620,490,570,33,26,navy,true);
text(b,'Anomaly evidence: reconstruction error\nat each timestep, averaged over sensors.',620,543,570,76,25);
text(b,'Default inference uses the encoder and decoder.',620,633,570,32,20,muted);
b.speakerNotes.text=`The user refers to a phase-one reconstructor. In the current code, phase one performs latent reconstruction to pretrain the encoder, and phase two trains the signal reconstruction head. This slide explains the encoder handoff rather than conflating these stages. Default: transfer encoder weights, initialize the reconstruction head randomly, freeze the encoder (including evaluation-mode statistics through trainer handling), and train the head on full normal windows without masking. Finetune mode is optional. The phase-one predictor is not part of the default reconstruction inference model. The auxiliary head is optional for a separate aux-crop variant, which this slide does not depict. The example uses the frequency_aux encoder config: channels [32,64,128], strides [2,2,2], total stride 8; C=38, W=512 gives latents (128,64). Mirrored transposed convolutions restore W=512 and a final 1x1 convolution restores C=38. Window lengths must divide evenly by the total encoder stride. Training minimizes mean L1 over all channels and timesteps. Default l1 scoring averages absolute residuals over channels at each timestep; score_mode can change the score reduction without changing L1 training. Sources: ${root}/lewm_reconstruction.py (_load_encoder_source, run_recon); ${root}/models/recon_head.py; ${root}/models/recon_model.py; ${root}/configs/lewm_encoder/_templates/reconstruction.yml; ${root}/configs/lewm_encoder/frequency_aux/phase1.yml; ${root}/utils/reconstruction_scores.py.`;

await fs.writeFile(path.join(build,'font-policy.json'),JSON.stringify({basis:'design',families:['Arial']}));
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidatePath=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(workspaceDir,'output/LeWM_Phase_One_and_Reconstructor.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,
 pythonExecutable:'/home/manos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:2,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));
