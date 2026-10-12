from ultralytics.utils.loss import v8DetectionLoss


class TrayDetectionLoss:
    def __init__(self, yolo_model):
        self.criterion = v8DetectionLoss(yolo_model)

    def __call__(self, predictions, targets):
        total, components = self.criterion(predictions, targets)
        return total.sum(), {name: value for name, value in zip(("box", "cls", "dfl"), components)}
