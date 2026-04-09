import torch.nn as nn
import torchvision.models as models
import config
import logging


def get_orientation_model(pretrained=True, num_blocks_to_unfreeze=None):
    """
    Loads a pre-trained EfficientNet model and configures it for fine-tuning.

    Args:
        pretrained (bool): Whether to load ImageNet weights.
        num_blocks_to_unfreeze (int | None): How many of the final 8 feature
            blocks to unfreeze. Uses config.NUM_BLOCKS_TO_UNFREEZE when None.
            Set to 8 to unfreeze all feature blocks (full fine-tuning).
    """
    if num_blocks_to_unfreeze is None:
        num_blocks_to_unfreeze = config.NUM_BLOCKS_TO_UNFREEZE

    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head first, which is always desirable.
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze the specified number of final blocks in the feature extractor
    if num_blocks_to_unfreeze > 0:
        # Slicing from a negative index unfreezes the last N blocks.
        for block in model.features[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

    logging.info(f"Fine-tuning enabled: Unfroze the final {num_blocks_to_unfreeze} feature blocks and the classifier.")

    # Get the number of input features for the classifier
    num_ftrs = model.classifier[1].in_features

    # Replace the final fully connected layer.
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, config.NUM_CLASSES),
    )

    return model
