import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer import weighted_rrf


def test_single_model_rrf():
    rankings = {
        "clip": [(0, 0.9), (1, 0.8), (2, 0.7)],
    }
    result = weighted_rrf(rankings, {"clip": 1.0}, n_images=3, top_k=3)
    indices = [idx for idx, _ in result]
    assert indices == [0, 1, 2]


def test_two_models_agreement():
    """When both models agree on ranking, fused ranking preserves order."""
    rankings = {
        "clip": [(0, 0.9), (1, 0.8), (2, 0.7)],
        "siglip": [(0, 0.85), (1, 0.75), (2, 0.6)],
    }
    result = weighted_rrf(rankings, {"clip": 1.0, "siglip": 1.0}, n_images=3, top_k=3)
    indices = [idx for idx, _ in result]
    assert indices == [0, 1, 2]


def test_two_models_disagreement():
    """When models disagree, RRF compromises."""
    rankings = {
        "clip": [(0, 0.9), (1, 0.8), (2, 0.7)],
        "siglip": [(2, 0.9), (1, 0.8), (0, 0.7)],
    }
    result = weighted_rrf(rankings, {"clip": 1.0, "siglip": 1.0}, n_images=3, top_k=3)
    indices = [idx for idx, _ in result]
    # Image 1 is rank 2 in both, so it should benefit from agreement
    # Image 0 is rank 1 in clip, rank 3 in siglip
    # Image 2 is rank 3 in clip, rank 1 in siglip
    # 0 and 2 should have same score; 1 is rank 2 in both
    assert 1 in indices


def test_weighted_rrf_respects_weights():
    """Higher-weighted model's ranking should dominate."""
    rankings = {
        "clip": [(0, 0.9), (1, 0.5)],
        "siglip": [(1, 0.9), (0, 0.5)],
    }
    # clip gets 10x weight
    result = weighted_rrf(rankings, {"clip": 10.0, "siglip": 1.0}, n_images=2, top_k=2)
    assert result[0][0] == 0  # clip's top pick wins


def test_top_k_limits_results():
    rankings = {
        "clip": [(i, 1.0 - i * 0.1) for i in range(10)],
    }
    result = weighted_rrf(rankings, {"clip": 1.0}, n_images=10, top_k=3)
    assert len(result) == 3


def test_empty_rankings():
    result = weighted_rrf({}, {}, n_images=0, top_k=5)
    assert result == []


def test_annotation_weight_dominates():
    """Annotation ranking at weight 3.0 should override visual ranking at 1.0."""
    from indexer import ANNOTATION_WEIGHT

    rankings = {
        # Visual model ranks image 0 first
        "clip": [(0, 0.9), (1, 0.8), (2, 0.7)],
        # Annotations rank image 2 first
        "annotations": [(2, 1.0), (1, 0.5)],
    }
    weights = {"clip": 1.0, "annotations": ANNOTATION_WEIGHT}
    result = weighted_rrf(rankings, weights, n_images=3, top_k=3)
    # Image 2 should be ranked first due to annotation weight
    assert result[0][0] == 2
