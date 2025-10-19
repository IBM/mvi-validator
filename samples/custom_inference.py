#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, override

import mvi_validator

class InferenceClient(mvi_validator.AbcInferenceClient):
    """
    Abstract base class for an inference client that performs model inference directly from raw image bytes.

    Implementations must define the `infer_bytes()` method, which accepts image data as bytes and
    returns the inference results in a standardized dictionary format.

    Expected return format:
        {
            "result": "success",               # "success" or "fail"
            "inference_sec": 1.23,             # float, inference execution time in seconds
            "classified": [
                {
                    "label": "donut",          # str, predicted class label
                    "confidence": 0.999977,    # float, confidence score (0.0–1.0)
                    "xmin": 449,               # int, bounding box top-left x coordinate
                    "ymin": 576,               # int, bounding box top-left y coordinate
                    "xmax": 817,               # int, bounding box bottom-right x coordinate
                    "ymax": 925                # int, bounding box bottom-right y coordinate
                },
                ...
            ]
        }
    """
    def __init__(self):
        self._id = "LocalInferenceClient_001"
    
    @override
    def id(self) -> str:
        return self._id

    @override
    def infer_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run inference on the given image bytes and return results
        as a structured dictionary.

        Args:
            image_bytes (bytes): Raw image data to infer.

        Returns:
            dict: Inference result dictionary in the required format.

        Raises:
            ValueError: If the input image cannot be decoded.
            RuntimeError: If the inference process fails.
        """

        result_json: Dict[str, Any] = {
            "result": "success",               # "success" or "fail"
            "inference_sec": 1.23,             # float, inference execution time in seconds
            "classified": [
                {
                "label": "dent",
                "confidence": 0.9999735355377197,
                "xmin": 610,
                "ymin": 1425,
                "xmax": 1154,
                "ymax": 2016
                },
                {
                "label": "dent",
                "confidence": 0.999957799911499,
                "xmin": 2562,
                "ymin": 1520,
                "xmax": 3021,
                "ymax": 1946
                }
            ]
        }

        return result_json
