


class DetectionLoss:

    def __init__(self, model):

        device = next(model.parameters()).device

        m = model.model[-1]
        self.num_classes = m.num_classes
        