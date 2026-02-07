from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from torch.nn import Linear
import torch.nn as nn
import torch


def make_analog_peft_compatible(analog_layer):
    """
    Make an AnalogLinear layer compatible with PEFT by providing a weight property.

    PEFT's LoRA implementation expects layers to have a .weight attribute
    that returns a tensor with .dtype. AnalogLinear stores weights in the
    analog tile, so we add a method to retrieve them.

    Args:
        analog_layer: An AnalogLinear instance to patch

    Returns:
        The same layer with PEFT compatibility added
    """
    # Store a reference to the weight tensor for PEFT compatibility
    # PEFT only needs .weight.dtype, so we create a parameter-like object
    weights = analog_layer.analog_module.tile.get_weights()
    # Store the actual weight tensor as a regular attribute (not a parameter)
    analog_layer._peft_weight = weights

    # Override the weight property using object attribute
    # This works because AnalogLinear sets self.weight = None after init
    analog_layer.weight = weights

    return analog_layer


def list_linear_layers(module, parent_name=''):
    """
    Recursively list all Linear layers in the module and return their names.
    """
    linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, Linear):  # Replace with the actual class name of Linear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            linear_layers.extend(child_layers)

    return linear_layers


def convert_to_analog_according_to_name(module, parent_name=''):
    """
    Recursively list all Linear layers in the module and return their names.
    """
    linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, Linear):  # Replace with the actual class name of Linear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            linear_layers.extend(child_layers)

    return linear_layers

def convert_selected_layers_to_analog(model, layers_to_convert, rup_conf):
    """
    Converts selected layers of the given model to analog, based on their names.
    """
    for name, module in model.named_modules():
        # if name in layers_to_convert:
        if any(substring in name for substring in layers_to_convert):
            # Assuming convert_to_analog is a function that takes a layer as input
            # and returns its analog equivalent
            setattr(model, name, convert_to_analog(module, rup_conf, tile_module_class=TorchInferenceTile))
            # TorchInferenceTile

    return model

def list_analog_linear_layers(module, parent_name=''):
    """
    Recursively list all AnalogLinear layers in the module and return their names.
    """
    analog_linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, AnalogLinear):  # Replace with the actual class name of AnalogLinear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            analog_linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_analog_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            analog_linear_layers.extend(child_layers)

    return analog_linear_layers



def replace_layer(model, digital_model, layer_name):
    """
    Replace a specific layer in the model with the corresponding layer from the digital model.
    """
    # Access the layer in the model and in the digital model using the layer name
    layer_hierarchy = layer_name.split('.')
    parent = model
    digital_parent = digital_model

    # Traverse to the parent of the layer
    for name in layer_hierarchy[:-1]:
        parent = getattr(parent, name)
        digital_parent = getattr(digital_parent, name)

    # Replace the layer
    setattr(parent, layer_hierarchy[-1], getattr(digital_parent, layer_hierarchy[-1]))


def get_parent_module(model, layer_name):
    """
    Get the parent module and attribute name for a given layer path.

    Args:
        model: The model to search in
        layer_name: Dot-separated path to the layer (e.g., 'encoder.layer.0.lora_A')

    Returns:
        Tuple of (parent_module, attribute_name)
    """
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_lora_layers_only_to_analog(model, rpu_config):
    """
    Convert only LoRA adapter layers (lora_A, lora_B) to analog.
    Base model layers are kept digital.

    This function identifies all Linear layers that are part of LoRA adapters
    (identified by 'lora_A' or 'lora_B' in their names) and converts them to
    AnalogLinear layers using the provided RPU configuration.

    Args:
        model: The model with LoRA adapters
        rpu_config: RPU configuration for analog conversion
            - TorchInferenceRPUConfig: Uses TorchInferenceTile (inference-only)
            - SingleRPUConfig/UnitCellRPUConfig: Uses default AnalogTile (training)

    Returns:
        The model with LoRA layers converted to analog
    """
    # Collect all LoRA layer names
    lora_layer_names = []
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, Linear):
            lora_layer_names.append(name)

    print(f"Found {len(lora_layer_names)} LoRA layers to convert to analog:")
    for name in lora_layer_names:
        print(f"  - {name}")

    # Determine tile_module_class based on config type
    # TorchInferenceRPUConfig requires TorchInferenceTile
    # SingleRPUConfig/UnitCellRPUConfig use default AnalogTile (no tile_module_class needed)
    use_inference_tile = isinstance(rpu_config, TorchInferenceRPUConfig)
    if use_inference_tile:
        print("Using TorchInferenceTile for TorchInferenceRPUConfig")
    else:
        print(f"Using default AnalogTile for {type(rpu_config).__name__}")

    # Convert each LoRA layer to analog and make PEFT-compatible
    for layer_name in lora_layer_names:
        parent, attr = get_parent_module(model, layer_name)
        original_layer = getattr(parent, attr)

        # Convert to analog layer with appropriate tile class
        if use_inference_tile:
            analog_layer = AnalogLinear.from_digital(
                original_layer,
                rpu_config,
                tile_module_class=TorchInferenceTile
            )
        else:
            analog_layer = AnalogLinear.from_digital(
                original_layer,
                rpu_config,
            )

        # Make PEFT-compatible by setting the weight attribute
        analog_layer = make_analog_peft_compatible(analog_layer)

        # Replace the layer
        setattr(parent, attr, analog_layer)
        print(f"Converted {layer_name} to AnalogLinear (PEFT-compatible)")

    return model


def list_lora_layers(module, parent_name=''):
    """
    Recursively list all LoRA adapter layers (lora_A, lora_B) in the module.

    Args:
        module: The module to search
        parent_name: Parent name for building full path

    Returns:
        List of layer names that are LoRA adapters
    """
    lora_layers = []
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if 'lora_A' in name or 'lora_B' in name:
            if isinstance(child, (Linear, AnalogLinear)):
                lora_layers.append(full_name)
        # Recursively apply to child modules
        child_layers = list_lora_layers(child, full_name)
        lora_layers.extend(child_layers)

    return lora_layers


def apply_sixt1c_retention(model, time_sec, tau_sec=46505.0):
    """
    Apply 6T1C retention decay to all AnalogLinear layers in the model.

    For sixt1c devices, retention follows exponential decay: w(t) = w(0) * exp(-t/tau)
    This function directly modifies the weights of AnalogLinear layers to simulate
    the effect of retention over the specified time.

    Args:
        model: The model containing AnalogLinear layers
        time_sec: Time in seconds to simulate retention decay
        tau_sec: Time constant (default: 46505.0 seconds)

    Returns:
        The model with retention decay applied to analog weights
    """
    import math
    import torch

    if time_sec <= 0:
        return model

    decay_factor = math.exp(-time_sec / tau_sec)
    print(f"Applying sixt1c retention: time={time_sec}s, decay_factor={decay_factor:.6f}")

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            # Get the analog tile
            tile = module.analog_module.tile
            # Get current weights (returns just a tensor, not a tuple)
            weights = tile.get_weights()
            # Apply exponential decay
            decayed_weights = weights * decay_factor
            # Set the decayed weights back
            tile.set_weights(decayed_weights)
            # Update the PEFT-compatible weight attribute
            if hasattr(module, 'weight'):
                module.weight = decayed_weights

    return model


def apply_sixt1c_lora_retention(model, time_sec, tau_sec=46505.0):
    """
    Apply 6T1C retention decay only to LoRA adapter layers (lora_A, lora_B).

    This function is used for sixt1c_lora mode where:
    - base_layer: AnalogLinear (PCM) - not affected by this function
    - lora_A/lora_B: AnalogLinear (Sixt1c) - retention decay applied

    For sixt1c devices, retention follows exponential decay: w(t) = w(0) * exp(-t/tau)

    Args:
        model: The model containing AnalogLinear layers
        time_sec: Time in seconds to simulate retention decay
        tau_sec: Time constant (default: 46505.0 seconds)

    Returns:
        The model with retention decay applied to LoRA analog weights only
    """
    import math

    if time_sec <= 0:
        return model

    decay_factor = math.exp(-time_sec / tau_sec)
    print(f"Applying sixt1c retention to LoRA layers: time={time_sec}s, decay_factor={decay_factor:.6f}")

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            # Only apply to LoRA layers (lora_A and lora_B), not base_layer
            if 'lora_A' in name or 'lora_B' in name:
                # Get the analog tile
                tile = module.analog_module.tile
                # Get current weights (returns just a tensor, not a tuple)
                weights = tile.get_weights()
                # Apply exponential decay
                decayed_weights = weights * decay_factor
                # Set the decayed weights back
                tile.set_weights(decayed_weights)
                # Update the PEFT-compatible weight attribute
                if hasattr(module, 'weight'):
                    module.weight = decayed_weights
                print(f"  Applied retention to {name}")

    return model


def apply_pcm_drift_to_base_layer(model, drift_time):
    """
    Apply PCM drift to base_layer only (not lora_A/lora_B).

    This function is used for sixt1c_lora mode where:
    - base_layer: AnalogLinear (PCM) - PCM drift applied
    - lora_A/lora_B: AnalogLinear (Sixt1c) - not affected by this function

    Args:
        model: The model containing AnalogLinear layers (must be in eval mode)
        drift_time: Time in seconds for drift simulation

    Note: drift_analog_weights() can be called on individual AnalogLinear modules.
    It uses the RPU noise_model to apply drift (requires InferenceRPUConfig).

    Returns:
        The model with PCM drift applied to base_layer weights
    """
    if drift_time <= 0:
        return model

    print(f"Applying PCM drift to base_layer: drift_time={drift_time}s")

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            # Only apply to base_layer (exclude lora_A and lora_B)
            if 'base_layer' in name:
                module.drift_analog_weights(drift_time)
                print(f"  Applied PCM drift to {name}")

    return model


def apply_tikitaka_drift(model, drift_time):
    """
    Apply drift to TikiTaka v2 Slow tiles (weight_tile).

    TikiTaka v2 uses a 2-tile structure with ChoppedTransferCompound:
    - Fast tile (grad_tile): 6T1C LinearStepDevice - receives SGD updates
    - Slow tile (weight_tile): SoftBoundsDevice - used in forward pass (gamma=0)

    This function applies drift only to the Slow tile (weight_tile), which is
    the tile visible during inference. The drift is applied using the
    drift_weights() method on the underlying tile.

    Args:
        model: The model containing AnalogLinear layers with TikiTaka v2 config
        drift_time: Time in seconds for drift simulation

    Returns:
        The model with drift applied to TikiTaka v2 Slow tiles
    """
    if drift_time <= 0:
        return model

    print(f"Applying TikiTaka drift: drift_time={drift_time}s")

    drift_count = 0
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            tile = module.analog_module.tile
            # TransferSimulatorTile (TikiTaka) has weight_tile attribute
            if hasattr(tile, 'weight_tile'):
                # Apply drift to the Slow tile (weight_tile)
                tile.weight_tile.tile.drift_weights(drift_time)
                drift_count += 1
                print(f"  Applied drift to {name}")

    print(f"  Total: Applied drift to {drift_count} TikiTaka layers")
    return model

