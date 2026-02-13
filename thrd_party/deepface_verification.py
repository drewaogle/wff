# built-in dependencies
import time
from typing import Any, Dict, Optional, Union, List, Tuple, IO, cast
import math

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

from thrd_party.deepface_config_confidence import confidences
from thrd_party.deepface_config_threshold import thresholds




def find_cosine_distance(
    source_representation: Union[NDArray[Any], List[float]],
    test_representation: Union[NDArray[Any], List[float]],
) -> Union[np.float64, NDArray[Any]]:
    """
    Find cosine distance between two given vectors or batches of vectors.
    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.
    Returns
        np.float64 or np.ndarray: Calculated cosine distance(s).
        It returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        distances = 1 - dot_product / (source_norm * test_norm)
        return cast(np.float64, distances)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        cosine_similarities = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = 1 - cosine_similarities
        return cast(NDArray[Any], distances)
    else:
        raise InvalidEmbeddingsShapeError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )


def find_angular_distance(
    source_representation: Union[NDArray[Any], List[float]],
    test_representation: Union[NDArray[Any], List[float]],
) -> Union[np.float64, NDArray[Any]]:
    """
    Find angular distance between two vectors or batches of vectors.

    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: angular distance(s).
            Returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """

    # calculate cosine similarity first
    # then convert to angular distance
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        similarity = dot_product / (source_norm * test_norm)
        distances = np.arccos(similarity) / np.pi
        return cast(np.float64, distances)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        similarity = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = np.arccos(similarity) / np.pi
        return cast(NDArray[Any], distances)
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )


def find_euclidean_distance(
    source_representation: Union[NDArray[Any], List[float]],
    test_representation: Union[NDArray[Any], List[float]],
) -> Union[np.float64, NDArray[Any]]:
    """
    Find Euclidean distance between two vectors or batches of vectors.

    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: Euclidean distance(s).
            Returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    # Single embedding case (1D arrays)
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        distances = np.linalg.norm(source_representation - test_representation)
        return cast(np.float64, distances)
    # Batch embeddings case (2D arrays)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        diff = (
            source_representation[None, :, :] - test_representation[:, None, :]
        )  # (N, D) - (M, D)  = (M, N, D)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
        return cast(NDArray[Any], distances)
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )


def l2_normalize(
    x: Union[NDArray[Any], List[float], List[List[float]]],
    axis: Union[int, None] = None,
    epsilon: float = 1e-10,
) -> NDArray[Any]:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
        axis (int): axis along which to normalize
    Returns:
        np.ndarray: l2 normalized vector
    """
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return cast(NDArray[Any], x / (norm + epsilon))


def find_distance(
    alpha_embedding: Union[NDArray[Any], List[float]],
    beta_embedding: Union[NDArray[Any], List[float]],
    distance_metric: str,
) -> Union[np.float64, NDArray[Any]]:
    """
    Wrapper to find the distance between vectors based on the specified distance metric.

    Args:
        alpha_embedding (np.ndarray or list): 1st vector or batch of vectors.
        beta_embedding (np.ndarray or list): 2nd vector or batch of vectors.
        distance_metric (str): The type of distance to compute
            ('cosine', 'euclidean', 'euclidean_l2', or 'angular').

    Returns:
        np.float64 or np.ndarray: The calculated distance(s).
    """
    # Convert inputs to numpy arrays if necessary
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    # Ensure that both embeddings are either 1D or 2D
    if alpha_embedding.ndim != beta_embedding.ndim or alpha_embedding.ndim not in (1, 2):
        raise ValueError(
            f"Both embeddings must be either 1D or 2D, but received "
            f"alpha shape: {alpha_embedding.shape}, beta shape: {beta_embedding.shape}"
        )

    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "angular":
        distance = find_angular_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        axis = None if alpha_embedding.ndim == 1 else 1
        normalized_alpha = l2_normalize(alpha_embedding, axis=axis)
        normalized_beta = l2_normalize(beta_embedding, axis=axis)
        distance = find_euclidean_distance(normalized_alpha, normalized_beta)
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return np.round(distance, 6)


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            euclidean_l2 and angular.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """
    if thresholds.get(model_name) is None:
        raise ValueError(f"Model {model_name} is not supported. ")

    threshold = thresholds.get(model_name, {}).get(distance_metric)

    if threshold is None:
        raise ValueError(
            f"Distance metric {distance_metric} is not available for model {model_name}. "
        )

    return threshold


def __sigmoid(z: float) -> float:
    """
    Compute a numerically stable sigmoid-based confidence score.

    This implementation avoids floating-point overflow errors that can occur
    when computing the standard sigmoid function (1 / (1 + exp(-z))) for very
    large positive or negative values of `z`. The computation is split based on
    the sign of `z` to ensure numerical stability while preserving mathematical
    equivalence.

    Args:
        z (float): Input value.

    Returns:
        float: Sigmoid output scaled to the range [0, 1].
    """
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        ez = math.exp(z)
        return 1 * ez / (1 + ez)


def find_confidence(
    distance: float, model_name: str, distance_metric: str, verified: bool
) -> float:
    """
    Using pre-built logistic regression model, find confidence value from distance.
        The confidence score provides a probalistic estimate, indicating how likely
        the classification is correct, thus giving softer, more informative measure of
        certainty than a simple binary classification.

        Configuration values are calculated in experiments/distance-to-confidence.ipynb
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            euclidean_l2 and angular.
        verified (bool): True if the images are classified as same person,
            False if different persons.
    Returns:
        confidence (float): confidence value being same person for that model name
            and distance metric pair. Same person classifications confidence should be
            distributed between 51-100% and different person classifications confidence
            should be distributed between 0-49%. The higher the confidence, the more
            certain the model is about the classification.
    """
    if distance <= 0:
        return 100.0 if verified else 0.0

    if confidences.get(model_name) is None:
        return 51 if verified else 49

    config = confidences[model_name].get(distance_metric)

    if config is None:
        return 51 if verified else 49

    w = config["w"]
    b = config["b"]

    normalizer = config["normalizer"]

    denorm_max_true = config["denorm_max_true"]
    denorm_min_true = config["denorm_min_true"]
    denorm_max_false = config["denorm_max_false"]
    denorm_min_false = config["denorm_min_false"]

    if normalizer > 1:
        distance = distance / normalizer

    z = w * distance + b
    confidence = 100 * __sigmoid(z)

    # re-distribute the confidence between 0-49 for different persons, 51-100 for same persons
    if verified:
        min_original = denorm_min_true
        max_original = denorm_max_true
        min_target = max(51, min_original)
        max_target = 100
    else:
        min_original = denorm_min_false
        max_original = denorm_max_false
        min_target = 0
        max_target = min(49, int(max_original))

    confidence_distributed = ((confidence - min_original) / (max_original - min_original)) * (
        max_target - min_target
    ) + min_target

    # ensure confidence is within 51-100 for same persons and 0-49 for different persons
    if verified and confidence_distributed < 51:
        confidence_distributed = 51
    elif not verified and confidence_distributed > 49:
        confidence_distributed = 49

    # ensure confidence is within 0-100
    if confidence_distributed < 0:
        confidence_distributed = 0
    elif confidence_distributed > 100:
        confidence_distributed = 100

    return round(confidence_distributed, 2)
