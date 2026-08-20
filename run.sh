CFG_ENV=configs/env.yml;
CFG_EXP=configs/pretext/new_loss/smd/original-dynamic_weight-loss_clamp/dynamic_weight_loss_using_negative_instances_clamp_loss.yml;
VER=batch/original-dynamic_weight-loss_clamp/dynamic_weight_loss_using_negative_instances_clamp_loss;
for M in svm svc iforest;
    do echo "=== METHOD $M ===";
    python carla_pretext_inference.py --config_env $CFG_ENV --config_exp $CFG_EXP --fname machine-2-5.txt --version $VER --method $M;
    done' > /tmp/opencode/all_methods_run.log 2>&1 &
echo "started pid $!"