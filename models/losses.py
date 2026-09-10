# Representation similarity scores for loss (CLIP) and retrieval
# Adapted from Meta's BrainMagick project:
# https://github.com/facebookresearch/brainmagick

import torch
import torch.nn.functional as F


class ClipLoss(torch.nn.Module):
    """CLIP-style contrastive loss."""

    def __init__(self):
        super().__init__()

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates [B, C, T] and candidates [B', C, T], return [B, B'] scores."""
        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate):
        """Contrastive training objective over a candidate batch."""
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)


class DClipLoss(ClipLoss):
    """CLIP loss robust to duplicate stimuli in a batch."""

    def forward(self, estimate, candidate, stim_ids=None):
        if stim_ids is None:
            return super().forward(estimate, candidate)

        assert estimate.size(0) == candidate.size(0), \
            "DClipLoss expects matched estimate and candidate batches"

        scores = self.get_scores(estimate, candidate)
        batch_size = scores.size(0)

        if isinstance(stim_ids, torch.Tensor):
            ids = stim_ids.tolist()
        else:
            ids = list(stim_ids)
        same_stim = torch.tensor(
            [[ids[i] == ids[j] for j in range(batch_size)] for i in range(batch_size)],
            dtype=torch.bool,
            device=estimate.device,
        )

        eye = torch.eye(batch_size, dtype=torch.bool, device=estimate.device)
        false_negatives = same_stim & ~eye
        scores = scores.masked_fill(false_negatives, float("-inf"))

        target = torch.arange(batch_size, device=estimate.device)
        return F.cross_entropy(scores, target)
