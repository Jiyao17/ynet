


SIZES = {
    'n': (0.33, 0.25, 2.0),
    's': (0.33, 0.50, 2.0),
    'm': (0.50, 0.67, 1.5),
    'l': (1.00, 1.00, 1.0),
    'x': (1.00, 1.25, 1.0),
    }


def check_size():
    net = FCOS(
        backbone=FCOSBackbone(*SIZES['x']),
        num_classes=3,
        anchor_generator=AnchorGenerator(
            sizes=((8,), (16,), (32,),),  # equal to strides of multi-level feature map
            aspect_ratios=((1.0,),) * 3, # equal to num_anchors_per_location
        ),
        score_thresh=0.2,
        nms_thresh=1e-5,
        detections_per_img=2,
        topk_candidates=64,
        )

    param_num = 0
    for param in net.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print("Total number of parameters: {} M".format(param_num / 1e6))


if __name__ == '__main__':
    check_size()