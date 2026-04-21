#!/usr/bin/env python3
"""Quick diagnostic: compare default pulse vs NoneWithDevice with large transfer_lr.

Piggybacks on the optuna script infrastructure but runs a single fixed-param trial.

Usage:
    CUDA_VISIBLE_DEVICES=3 python diag_transfer_lr.py default
    CUDA_VISIBLE_DEVICES=3 python diag_transfer_lr.py nwd
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import optuna

# Import everything from the optuna script
import optuna_bert_squad_lrtt as O

PULSE_TYPE = sys.argv[1] if len(sys.argv) > 1 else "default"
DW_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001981

# Configure globals to match the experiment
O.BATCH_SIZE = 48
O.GRAD_ACCUM_STEPS = 2  # Match OOM retry condition
O.N_EPOCHS = 1
O.WARMUP_STEPS = 365
O.TRANSFER_METHOD = "set"
O.AB_DEVICE = "linearstep"
O.C_DEVICE = "ideal"
O.IO_NOISE = False
O.FORWARD_INJECT = True
O.IS_PERFECT = True
O.NO_QUANT = False
O.LORA_TARGET = "qkvo"
O.HEAD_LAYER = False
O.ENCODER_ANALOG = True
O.HEAD_ANALOG = True
O.BACKWARD_OUT_BOUND = 12.0
O.REINIT_GAIN = 0.01
O.SEED = 42
O.TRAIN_SUBSET_SIZE = 0
O.DYNAMIC_TE = False
O.DYNAMIC_TE_POWER = 1.0
O.TE_WARMUP_SCHEDULE = []
O.TE_WARMUP_STEPS = 0

O.OPT_CONFIG = {
    'optimizer': 'AnalogAdam',
    'reinit_mode': 'hybrid',
    'tune_wd': False,
    'tune_momentum': False,
    'tune_nesterov': False,
    'no_transfer': False,
    'learn_out_scaling': False,
    'correct_gradient_magnitudes': False,
    'no_adc_ab_proj': False,
    'auto_scale_mode': 'none',
    'scale_transfer_lr': True,
    'fi_continuous_alpha': True,
    'ab_pulse_type': 'none_with_device' if PULSE_TYPE == 'nwd' else 'default',
    'transfer_rank_schedule': 'all',
    'transfer_ranks_per_step': 1,
}

# Exact params from linearstep NaN Trial 3 (F1=0, NaN abort)
FIXED_PARAMS = {
    'learning_rate': 0.004125,
    'transfer_lr': 639.0,
    'transfer_every': 5,
    'rank_exp': 1,          # rank=2
    'fast_lr': 0.1428,
    'tau_sec': 0.0,
    'ab_dw_min': 0.1262,
    'ab_desired_bl': 31,
    'out_noise': 0.0,
    'ab_weight_scaling_omega': 0.0,
    'min_lr_rate': 0.0,
}


def main():
    pulse_str = "none_with_device" if PULSE_TYPE == "nwd" else "default"
    print(f"\n{'='*70}")
    print(f"  Diagnostic: ab_pulse_type = {pulse_str}")
    print(f"  transfer_lr={FIXED_PARAMS['transfer_lr']}, rank=2, fast_lr={FIXED_PARAMS['fast_lr']}")
    print(f"  ab_dw_min={FIXED_PARAMS['ab_dw_min']}, 1 epoch only")
    print(f"{'='*70}\n")

    # Load data
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(O.MODEL_NAME)
    train_loader, eval_features, eval_examples = O.load_data(tokenizer)

    # Create study with enqueued fixed trial
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(FIXED_PARAMS)

    study.optimize(
        lambda trial: O.objective(trial, train_loader, eval_features, eval_examples, tokenizer),
        n_trials=1,
    )

    print(f"\n  Result: F1 = {study.best_value:.2f}%")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
