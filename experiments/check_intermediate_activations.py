"""
Check activations at each stage from encoder to classifier.
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer


def check_activations(mode_name, fp_lora):
    """Check activations at each stage."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - ACTIVATION CHECK")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print(f"\n  Creating model (fp_lora={fp_lora})...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=fp_lora, lora_alpha=1.0)

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    # Hook to capture activations at each stage
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.max().item(),
                    'min': output.min().item(),
                    'max_abs': output.abs().max().item(),
                    'shape': tuple(output.shape),
                }
            elif isinstance(output, tuple) and len(output) > 0:
                # Take first element if output is a tuple
                out = output[0]
                activations[name] = {
                    'mean': out.mean().item(),
                    'std': out.std().item(),
                    'max': out.max().item(),
                    'min': out.min().item(),
                    'max_abs': out.abs().max().item(),
                    'shape': tuple(out.shape),
                }
        return hook

    # Register hooks
    hooks = []

    # Embeddings
    if hasattr(model.base_model.model, 'mobilebert'):
        if hasattr(model.base_model.model.mobilebert, 'embeddings'):
            hook = model.base_model.model.mobilebert.embeddings.register_forward_hook(make_hook('embeddings'))
            hooks.append(hook)

        # Encoder
        if hasattr(model.base_model.model.mobilebert, 'encoder'):
            hook = model.base_model.model.mobilebert.encoder.register_forward_hook(make_hook('encoder'))
            hooks.append(hook)

    # Pooler
    if hasattr(model.base_model.model, 'mobilebert'):
        if hasattr(model.base_model.model.mobilebert, 'pooler'):
            hook = model.base_model.model.mobilebert.pooler.register_forward_hook(make_hook('pooler'))
            hooks.append(hook)

    # Classifier
    if hasattr(model.base_model.model, 'classifier'):
        # Hook the input to classifier
        def classifier_input_hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0]
                activations['classifier_input'] = {
                    'mean': inp.mean().item(),
                    'std': inp.std().item(),
                    'max': inp.max().item(),
                    'min': inp.min().item(),
                    'max_abs': inp.abs().max().item(),
                    'shape': tuple(inp.shape),
                }
        hook = model.base_model.model.classifier.register_forward_hook(classifier_input_hook)
        hooks.append(hook)

        # Hook classifier output
        hook = model.base_model.model.classifier.register_forward_hook(make_hook('classifier_output'))
        hooks.append(hook)

    # Forward pass
    print(f"\n  Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print results
    print(f"\n  Activation Statistics:")
    print(f"  {'='*80}")

    stages = ['embeddings', 'encoder', 'pooler', 'classifier_input', 'classifier_output']
    for stage in stages:
        if stage in activations:
            stats = activations[stage]
            print(f"\n  {stage}:")
            print(f"    Shape: {stats['shape']}")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std: {stats['std']:.6f}")
            print(f"    Min: {stats['min']:.6f}")
            print(f"    Max: {stats['max']:.6f}")
            print(f"    Max Abs: {stats['max_abs']:.6f}")

            # Check for explosion
            if stats['max_abs'] > 1000:
                print(f"    ⚠️  EXPLOSION DETECTED! Values are > 1000")
        else:
            print(f"\n  {stage}: Not captured")

    print(f"\n  Final logits: {logits.squeeze().tolist()}")

    # Check classifier weights
    print(f"\n  Classifier Weights:")
    if hasattr(model.base_model.model, 'classifier'):
        classifier = model.base_model.model.classifier
        print(f"    Weight shape: {classifier.weight.shape}")
        print(f"    Weight mean: {classifier.weight.mean().item():.6f}")
        print(f"    Weight std: {classifier.weight.std().item():.6f}")
        print(f"    Weight max abs: {classifier.weight.abs().max().item():.6f}")
        if classifier.bias is not None:
            print(f"    Bias shape: {classifier.bias.shape}")
            print(f"    Bias mean: {classifier.bias.mean().item():.6f}")
            print(f"    Bias max abs: {classifier.bias.abs().max().item():.6f}")


def main():
    print("="*80)
    print("  INTERMEDIATE ACTIVATION INVESTIGATION")
    print("="*80)

    check_activations("SIXT1C", fp_lora=False)

    print("\n\n")

    check_activations("FP", fp_lora=True)

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
