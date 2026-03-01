# -*- coding: utf-8 -*-
"""Quick test: backward is_perfect=True — does loss change?"""
import os, sys, torch, time
os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, '/data')

import optuna_mobilebert_glue_lora as M
from aihwkit.optim import AnalogSGD
from transformers import AutoTokenizer

DEVICE = torch.device("cuda")
FIXED_PARAMS = {
    "rank": 8, "transfer_every": 10000000, "transfer_lr": 0.1,
    "lora_alpha": 0.01, "reinit_mode": "decay", "tau_sec": 0.0,
}
LR = 0.5
TARGET_AB_LR = 0.03

def setup():
    M.SEED = 42; M.MODEL_NAME = "google/mobilebert-uncased"; M.DEVICE = DEVICE
    M.TASK_NAME = "mrpc"; M.TASK_TO_NUM_LABELS = {"mrpc": 2}
    M.AB_DEVICE = "6t1c"; M.CONVERT_NONTARGET = True; M.HEAD_LAYER = "train"
    M.IO_NOISE = True; M.LORA_TARGET = "qkv"; M.BATCH_SIZE = 32
    M.N_EPOCHS = 5; M.MAX_SEQ_LENGTH = 128
    M.TRAIN_SUBSET_SIZE = 0; M.EVAL_SUBSET_SIZE = 0; M.WARMUP_RATIO = 0.1

if __name__ == "__main__":
    setup()
    M.set_seed(M.SEED)
    tokenizer = AutoTokenizer.from_pretrained(M.MODEL_NAME)
    train_loader, eval_loader = M.load_data(tokenizer)

    model = M.create_model(FIXED_PARAMS)

    # Verify backward is_perfect
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            am = module.analog_module
            if hasattr(am, 'tile_c'):
                # LRTT
                rpu = am.tile_c.rpu_config if hasattr(am.tile_c, 'rpu_config') else None
                if rpu and hasattr(rpu, 'backward'):
                    print(f"LRTT tile_c backward: {rpu.backward}")
                    break
            elif hasattr(am, 'rpu_config'):
                rpu = am.rpu_config
                if hasattr(rpu, 'backward'):
                    print(f"NT backward: {rpu.backward}")
                    break

    # Check initial loss
    model.eval()
    batch = next(iter(train_loader))
    inputs = {k: v.to(DEVICE) for k, v in batch.items()
              if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
    with torch.no_grad():
        out = model(**inputs)
        print(f"\nInitial loss: {out.loss.item():.4f}")
        print(f"Initial logits: [{out.logits.min().item():.1f}, {out.logits.max().item():.1f}]")

    # Setup optimizer
    lora_alpha = FIXED_PARAMS["lora_alpha"]
    lrtt_lr = TARGET_AB_LR / lora_alpha

    lrtt_tile_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'tile_a' in name or 'tile_b' in name:
                lrtt_tile_params.append(param)
            elif 'tile_c' in name and 'analog_ctx' in name:
                lrtt_tile_params.append(param)
            else:
                other_params.append(param)

    optimizer = AnalogSGD(
        [{'params': lrtt_tile_params, 'lr': lrtt_lr},
         {'params': other_params, 'lr': LR}],
        lr=LR, weight_decay=0.0, momentum=0.0,
    )
    optimizer.regroup_param_groups()

    lrtt_tile_ids = set()
    for m in model.modules():
        if hasattr(m, 'tile_a'):
            lrtt_tile_ids.add(id(m.tile_a))
            lrtt_tile_ids.add(id(m.tile_b))
            lrtt_tile_ids.add(id(m.tile_c))
    for group in optimizer.param_groups:
        for p in group["params"]:
            if hasattr(p, 'analog_tile') and id(p.analog_tile) in lrtt_tile_ids:
                group["lr"] = lrtt_lr
                p.analog_tile.set_learning_rate(lrtt_lr)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if hasattr(p, 'analog_tile') and id(p.analog_tile) not in lrtt_tile_ids:
                group["lr"] = 0.0
                p.analog_tile.set_learning_rate(0.0)

    tile_c_ctxs, tile_ab_ctxs = [], []
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_c'):
            t = module.analog_module
            if t.tile_c.analog_ctx is not None:
                tile_c_ctxs.append(t.tile_c.analog_ctx)
            for st in [t.tile_a, t.tile_b]:
                ctx = getattr(st, 'analog_ctx', None)
                if ctx is not None:
                    tile_ab_ctxs.append(ctx)

    num_steps = len(train_loader) * 5
    warmup_steps = int(num_steps * 0.1)
    scheduler = M.get_linear_schedule_with_min_lr(optimizer, warmup_steps, num_steps, min_lr_rate=0.0)

    # Train 2 epochs
    model.train()
    print(f"\nTraining with backward=is_perfect=True, LR={LR}, LRTT_LR={lrtt_lr}")
    for epoch in range(3):
        epoch_loss = 0.0
        n = 0
        t0 = time.time()
        for batch in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            optimizer.zero_grad()
            out = model(**inputs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for ctx in tile_c_ctxs:
                ctx.reset()
            for ctx in tile_ab_ctxs:
                ctx.reset()
            epoch_loss += out.loss.item()
            n += 1

        avg = epoch_loss / n
        metric, eval_loss = M.evaluate_model(model, eval_loader)
        print(f"  Epoch {epoch}: train_loss={avg:.4f}, eval_f1={metric:.4f}, time={time.time()-t0:.1f}s")
