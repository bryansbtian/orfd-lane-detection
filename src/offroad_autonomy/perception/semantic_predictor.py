"""Keep pixel confidence alongside Ultralytics' semantic class map."""

import torch
import torch.nn.functional as F
from ultralytics.engine.results import Results
from ultralytics.models.yolo.semantic.predict import SemanticSegmentationPredictor
from ultralytics.utils import ops


class RoadSemanticPredictor(SemanticSegmentationPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for i, (logits, original) in enumerate(zip(preds, orig_imgs)):
            if logits.ndim != 3:
                raise ValueError(
                    "Road semantic inference requires logits, not an exported class map"
                )
            # Restore the letterboxed input size before removing its padding.
            # This keeps low-resolution logits aligned with the original frame.
            logits = F.interpolate(
                logits[None].float(), img.shape[2:], mode="bilinear", align_corners=False
            )
            logits = ops.scale_masks(logits, original.shape[:2])[0]
            if logits.shape[0] == 1:
                probability = logits[0].sigmoid()
                labels = (probability > 0.5).long()
                confidence = torch.where(labels.bool(), probability, 1 - probability)
            else:
                confidence, labels = logits.softmax(0).max(0)
            path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            result = Results(original, path=path, names=self.model.names, semantic_mask=labels)
            # Upstream exposes only hard labels. Retain real probabilities for
            # the planner's confidence gate and the dashboard readout.
            result.semantic_confidence = confidence
            results.append(result)
        return results
