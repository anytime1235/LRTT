# Calculate grid search size
RANKS = [1, 4, 8, 16, 32, 64]
TRANSFER_EVERYS = [1, 10, 50, 100, 500, 1000, 2000, 5000]
LIFETIMES = [100, 1000, 10000, 46505, 100000]

# For grid search, we need discrete LR values
LRS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
TLRS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

total_trials = len(RANKS) * len(TRANSFER_EVERYS) * len(LRS) * len(TLRS) * len(LIFETIMES)
print(f"Total grid search trials: {total_trials}")
print(f"  RANKS: {len(RANKS)}")
print(f"  TRANSFER_EVERYS: {len(TRANSFER_EVERYS)}")
print(f"  LRS: {len(LRS)}")
print(f"  TLRS: {len(TLRS)}")
print(f"  LIFETIMES: {len(LIFETIMES)}")
print(f"  Total: {len(RANKS)} × {len(TRANSFER_EVERYS)} × {len(LRS)} × {len(TLRS)} × {len(LIFETIMES)} = {total_trials}")
