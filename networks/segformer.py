import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)


# reconstruct segformer in shadow detection
def segformer(pretrained=True):
    # id2label = {0: "others"}
    # label2id = {label: id for id, label in id2label.items()}
    # num_labels = len(id2label)
    # if pretrained:
    #     model = SegformerForSemanticSegmentation.from_pretrained(
    #         "/home/xingzhaohu/jiuding_code/mirror_adapter_prompt_depth/weight/segformer-b3-finetuned-ade-512-512",
    #         ignore_mismatched_sizes=True,
    #         num_labels=5,
    #         )
    #     return model

    # config = SegformerConfig.from_json_file("./config.json")
    config = SegformerConfig.from_json_file("./b0_config.json")
    config.num_labels = 151
    
    model = SegformerForSemanticSegmentation(config)
    return model 

import torch.nn as nn 

class SegFormer(nn.Module):
    def __init__(self, pretrianed=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = segformer(pretrianed)

    
    def forward(self, x):
        if len(x.shape) == 5:
            x = x[:, 1]
        logits = self.model(x).logits
        upsampled_logits = nn.functional.interpolate(
                logits, scale_factor=4.0, mode="bilinear", align_corners=False
            )
        return upsampled_logits


if __name__ == '__main__':
    from transformers import SegformerFeatureExtractor, SegformerForImageClassification


    model = SegformerForImageClassification.from_pretrained("/home/haipeng/Code/shadow/hgf_pretrain/nvidia/mit-b0")

    inputs = torch.randn(4,3,256,256)
    outputs = model(inputs,output_hidden_states=True)
    print("adasd")