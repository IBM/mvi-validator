#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================================
# mvi-validator
#
# Copyright (c) 2022 Takahide Nogayama
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# =================================================================

__version__ = "0.0.14"

import argparse
from argparse import ArgumentParser, Action, Namespace
import concurrent.futures
from collections import defaultdict
import io
import json
import glob
import logging
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET

import requests
import pandas as pd

# logger
_LOGGER: logging.Logger = logging.getLogger(__name__)

LOG_FORMAT: str = '%(asctime)s |  %(levelname)-7s | %(message)s (%(filename)s L%(lineno)s %(name)s)'

INCLUSION_RATE_THREASHOLD = 0.5
DEFAULT_NUM_OF_THREADS = 16
DEFAULT_PRINT_FORMAT = "markdown"
DEFAULT_SORT_KEYS = ["label"]

################################################################################
# MVI BoundingBox


class BoundingBox(object):

    def __init__(self, **entries):
        self.label = None
        self.iou: float = 0
        self.confidence: float = 0
        self.related_gt_bbox = None
        self.related_pd_bbox = None

        self.__dict__.update(entries)

        if self.xmin > self.xmax:
            raise ValueError(f"xmin: {self.xmin} is greater than xmax: {self.xmax}")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin: {self.ymin} is greater than ymax: {self.ymax}")

        self.width: int = self.xmax - self.xmin
        self.height: int = self.ymax - self.ymin
        self.area: int = self.width * self.height

        self.xcenter: float = self.xmin + (self.xmax - self.xmin) / 2
        self.ycenter: float = self.ymin + (self.ymax - self.ymin) / 2


    def intersection(self, that):
        ret_xmin = self.xmin if self.xmin > that.xmin else that.xmin
        ret_ymin = self.ymin if self.ymin > that.ymin else that.ymin
        ret_xmax = self.xmax if self.xmax < that.xmax else that.xmax
        ret_ymax = self.ymax if self.ymax < that.ymax else that.ymax
        if ret_xmin > ret_xmax or ret_ymin > ret_ymax:
            return BoundingBox(xmin=ret_xmin, ymin=ret_ymin, xmax=ret_xmin, ymax=ret_ymin, label=f"{self.label}&{that.label} (no intersection)")
        
        return BoundingBox(xmin=ret_xmin, ymin=ret_ymin, xmax=ret_xmax, ymax=ret_ymax, label=f"{self.label}&{that.label}")

    __and__ = intersection

    def get_inclusion_rate_by(self, that):
        intersection = self.intersection(that)
        return intersection.area / float(self.area)

    def is_included_by(self, that):
        if self.get_inclusion_rate_by(that) > INCLUSION_RATE_THREASHOLD:
            return True
        else:
            return False

    def get_union_area(self, that):
        intersection = self & that
        # _LOGGER.debug("self: %s", self)
        # _LOGGER.debug("that: %s", that)
        # _LOGGER.debug("intr: %s", intersection)
        # _LOGGER.debug("union: %s", self.area + that.area - intersection.area)
        return self.area + that.area - intersection.area

    def __repr__(self):
        return f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax}, xcenter={self.xcenter}, ycenter={self.ycenter}, label={self.label}, iou={self.iou}, confidence={self.confidence})"


################################################################################
# MVI Image XML

# <annotation>
#     <size>
#         <width>416</width>
#         <height>416</height>
#         <depth>3</depth>
#     </size>
#     <object>
#         <_id>95eaec23-5469-4aba-a5b3-67c84961e841</_id>
#         <name>Head</name>
#         <bndbox>
#             <xmin>193</xmin>
#             <ymin>93</ymin>
#             <xmax>252</xmax>
#             <ymax>173</ymax>
#         </bndbox>
#         <generate_type>manual</generate_type>
#         <file_id>c506ab6b-8dcc-48a2-88fe-bf8bd7897faa</file_id>
#     </object>
#     <object>
#         <_id>4fe1e398-c80c-4973-b8b1-e4c2cd24c74a</_id>
#         <name>Head</name>
#         <bndbox>
#             <xmin>82</xmin>
#             <ymin>121</ymin>
#             <xmax>151</xmax>
#             <ymax>207</ymax>
#         </bndbox>
#         <generate_type>manual</generate_type>
#         <file_id>c506ab6b-8dcc-48a2-88fe-bf8bd7897faa</file_id>
#     </object>
#     <object>
#         <_id>a607c0b2-c5e7-4504-b341-d68eb802ae5f</_id>
#         <name>Helmet</name>
#         <bndbox>
#             <xmin>193</xmin>
#             <ymin>93</ymin>
#             <xmax>251</xmax>
#             <ymax>136</ymax>
#         </bndbox>
#         <generate_type>manual</generate_type>
#         <file_id>c506ab6b-8dcc-48a2-88fe-bf8bd7897faa</file_id>
#     </object>
#     <object>
#         <_id>9e045fa2-f399-4496-b4c5-e264722e1699</_id>
#         <name>Helmet</name>
#         <bndbox>
#             <xmin>83</xmin>
#             <ymin>121</ymin>
#             <xmax>151</xmax>
#             <ymax>170</ymax>
#         </bndbox>
#         <generate_type>manual</generate_type>
#         <file_id>c506ab6b-8dcc-48a2-88fe-bf8bd7897faa</file_id>
#     </object>
# </annotation>

# When polygons
# <annotation>
#     <size>
#         <width>1086</width><height>1358</height><depth>3</depth>
#     </size>
#     <object>
#         <_id>23d12044-e520-47bb-b978-c2c874aa21a1</_id>
#         <name>donut</name>
#         <bndbox>
#             <xmin>476</xmin>
#             <ymin>588</ymin>
#             <xmax>784</xmax>
#             <ymax>896</ymax>
#         </bndbox>
#         <generate_type>manual</generate_type>
#         <file_id>0f2fbda5-daad-4786-9125-14d09553fbe2</file_id>
#         <segment_polygons>
#             <polygon>
#                 <point><value>631</value><value>588</value></point>
#                 <point><value>718</value><value>605</value></point>
#                 <point><value>771</value><value>673</value></point>
#                 <point><value>784</value><value>737</value></point>
#                 <point><value>767</value><value>808</value></point>
#                 <point><value>719</value><value>861</value></point>
#                 <point><value>678</value><value>884</value></point>
#                 <point><value>636</value><value>896</value></point>
#                 <point><value>578</value><value>883</value></point>
#                 <point><value>534</value><value>860</value></point>
#                 <point><value>501</value><value>823</value></point>
#                 <point><value>484</value><value>801</value></point>
#                 <point><value>476</value><value>751</value></point>
#                 <point><value>489</value><value>698</value></point>
#                 <point><value>521</value><value>646</value></point>
#                 <point><value>559</value><value>613</value></point>
#                 <point><value>590</value><value>595</value></point>
#             </polygon>
#         </segment_polygons>
#     </object>
# </annotation>


def load_image_metadata_from_xml(image_xml):
    gt_bboxes = []
    tree = ET.parse(image_xml)
    root = tree.getroot()
    for object_elem in root.iter("object"):
        name = object_elem.find("name").text
        bndbox_elem = object_elem.find("bndbox")
        xmin = int(bndbox_elem.find("xmin").text)
        ymin = int(bndbox_elem.find("ymin").text)
        xmax = int(bndbox_elem.find("xmax").text)
        ymax = int(bndbox_elem.find("ymax").text)
        if object_elem.find("segment_polygons"):
            _LOGGER.warning("Using Bounding Box instead of Polygons for grandtruth annotation. The calculated values is not highly accurate.")
        gt_bboxes.append(BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=name))
    return gt_bboxes


################################################################################
# MVI Inference JSON

# {
#     "webAPIId": "b841070a-e8cf-4b4d-872a-17bba83563ce",
#     "imageUrl": "http://masdemo-service:9080/masdemo-api/uploads/temp/b841070a-e8cf-4b4d-872a-17bba83563ce/46fd06c1-9757-4379-8157-545b82c988c3.jpg",
#     "imageMd5": "62e92dff7811aa7f0d6107724b9226f2",
#     "classified": [
#         {
#             "confidence": 0.9999966621398926,
#             "ymax": 173,
#             "label": "Head",
#             "xmax": 256,
#             "xmin": 189,
#             "ymin": 95
#         },
#         {
#             "confidence": 0.9999995231628418,
#             "ymax": 208,
#             "label": "Head",
#             "xmax": 154,
#             "xmin": 87,
#             "ymin": 124
#         },
#         {
#             "confidence": 0.9804211854934692,
#             "ymax": 167,
#             "label": "Helmet",
#             "xmax": 151,
#             "xmin": 89,
#             "ymin": 119
#         },
#         {
#             "confidence": 0.9988821148872375,
#             "ymax": 134,
#             "label": "Helmet",
#             "xmax": 254,
#             "xmin": 195,
#             "ymin": 92
#         },
#         {
#             "confidence": 0.9989311099052429,
#             "ymax": 42,
#             "label": "Mask",
#             "xmax": 247,
#             "xmin": 194,
#             "ymin": 2
#         }
#     ],
#     "result": "success"
# }

# When polygon
# {
#   "webAPIId": "2ace623e-31b6-4d8b-9d68-8f177e190c77",
#   "imageUrl": "https://masgeo-service:9080/masgeo-api/uploads/temp/2ace623e-31b6-4d8b-9d68-8f177e190c77/bd37f4d8-1656-41af-ae01-02e1a1c76576.jpg",
#   "classified": [
#     {
#       "label": "donut",
#       "confidence": 0.9999771118164062,
#       "xmin": 449,
#       "ymin": 576,
#       "xmax": 817,
#       "ymax": 925,
#       "polygons": [
#           [
#             [742,580],[741,581],[735,581],[734,582],[716,582],[715,583],[709,583],[708,584],[694,584],[693,585],
#             [679,585],[678,586],[667,586],[666,587],[653,587],[652,588],[639,588],[638,589],[628,589],[627,590],
#             [612,590],[611,591],[601,591],[600,592],[571,592],[570,591],[560,591],[559,590],[555,590],[554,589],
#             [549,589],[548,588],[534,588],[533,587],[530,587],[529,586],[527,586],[526,585],[523,585],[522,584],
#             [518,584],[517,585],[512,585],[511,586],[508,586],[507,587],[505,587],[504,588],[501,588],[500,589],
#             [498,589],[497,590],[495,590],[494,591],[493,591],[492,592],[491,592],[490,593],[489,593],[486,596],
#             [485,596],[484,597],[482,597],[481,598],[480,598],[479,599],[477,599],[476,600],[475,600],[474,601],
#             [473,601],[471,603],[470,603],[469,604],[468,604],[467,605],[466,605],[465,606],[464,606],[456,614],
#             [456,615],[455,616],[455,618],[454,619],[454,622],[453,623],[453,629],[452,630],[452,640],[451,641],
#             [451,655],[452,656],[452,658],[451,659],[451,680],[452,681],[452,686],[451,687],[451,788],[450,789],
#             [450,804],[451,805],[451,807],[450,808],[450,830],[451,831],[451,833],[450,834],[450,855],[451,856],
#             [451,910],[452,911],[452,919],[456,923],[466,923],[467,924],[493,924],[494,925],[637,925],[638,924],
#             [684,924],[685,923],[700,923],[701,922],[715,922],[716,921],[723,921],[724,920],[728,920],[729,919],
#             [736,919],[737,920],[743,920],[744,921],[747,921],[748,920],[758,920],[759,919],[762,919],[764,917],
#             [765,917],[769,913],[770,913],[771,912],[772,912],[774,910],[775,910],[776,909],[777,909],[779,907],
#             [780,907],[781,906],[782,906],[783,905],[784,905],[785,904],[786,904],[787,903],[789,903],[790,902],
#             [793,902],[794,901],[797,901],[798,900],[803,900],[804,899],[808,899],[809,898],[810,898],[811,897],
#             [811,896],[812,895],[812,891],[813,890],[813,877],[814,876],[814,853],[815,852],[815,674],[814,673],
#             [814,648],[815,647],[815,643],[814,642],[814,607],[813,606],[813,602],[812,601],[812,597],[811,596],
#             [811,591],[810,590],[810,587],[809,586],[807,586],[806,585],[798,585],[797,584],[796,584],[795,583],
#             [792,583],[791,582],[787,582],[786,581],[770,581],[769,580]
#           ]
#       ]
#     }
#   ],
#   "result": "success",
#   "saveInferenceResults": null
# }


def load_image_metadata_from_json(image_json):
    if isinstance(image_json, str):
        image_json = json.loads(image_json)

    pd_bboxes = []
    for classified in image_json.get("classified", []):
        label = classified["label"]
        xmin = classified["xmin"]
        ymin = classified["ymin"]
        xmax = classified["xmax"]
        ymax = classified["ymax"]
        confidence = classified["confidence"]
        if "polygons" in classified:
            _LOGGER.warning("Using Bounding Box instead of Polygons for prediction. The calculated values is not highly accurate.")
        pd_bboxes.append(BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label, confidence=confidence))
    return pd_bboxes


################################################################################


class ImageInfo(object):

    def __init__(self, image_stem, image_file, gt_bboxes=[], pd_bboxes=[]):
        self.image_stem = image_stem
        self.image_file = image_file
        self.gt_bboxes = gt_bboxes
        self.pd_bboxes = pd_bboxes

    def __repr__(self):
        return f"ImageInfo(image_file={self.image_file}, gt_bboxes={self.gt_bboxes}, pd_bboxes={self.pd_bboxes})"

    def analyze_iou(self):

        for pd_bbox in self.pd_bboxes:

            # Find the grand truth bbox that covers this predicted bbox maximally
            max_overlapped_gt_bbox: BoundingBox = None
            max_intersection_area: int = 0
            for gt_bbox in self.gt_bboxes:
                if gt_bbox.label == pd_bbox.label:
                    intersection = pd_bbox & gt_bbox
                    if not max_overlapped_gt_bbox \
                    or intersection.area > max_intersection_area:
                        max_overlapped_gt_bbox = gt_bbox
                        max_intersection_area = intersection.area

            if max_overlapped_gt_bbox:
                union_area = pd_bbox.get_union_area(max_overlapped_gt_bbox)
                pd_bbox.iou = max_intersection_area / union_area
                pd_bbox.related_gt_bbox = max_overlapped_gt_bbox
            else:
                pd_bbox.iou = 0.0

            _LOGGER.debug("Image:%s, pd_bbox object:   %s, iou: %s, confidence: %s, max_intersection_area: %s, union_area: %s", os.path.basename(self.image_file), pd_bbox.label, pd_bbox.iou, pd_bbox.confidence, max_intersection_area, union_area)

        for gt_bbox in self.gt_bboxes:

            # Find the predicted bbox that covers this grand truth bbox maximally
            max_overlapped_pd_bbox: BoundingBox = None
            max_intersection_area: int = 0
            for pd_bbox in self.pd_bboxes:
                if pd_bbox.label == gt_bbox.label:
                    intersection = gt_bbox & pd_bbox
                    if not max_overlapped_pd_bbox \
                    or intersection.area > max_intersection_area:
                        max_overlapped_pd_bbox = pd_bbox
                        max_intersection_area = intersection.area

            if max_overlapped_pd_bbox:
                union_area = gt_bbox.get_union_area(max_overlapped_pd_bbox)
                gt_bbox.iou = max_intersection_area / union_area
                gt_bbox.related_pd_bbox = max_overlapped_pd_bbox
            else:
                gt_bbox.iou = 0.0

            _LOGGER.debug("Image:%s, gt_bbox object: %s, iou: %s", os.path.basename(self.image_file), gt_bbox.label, gt_bbox.iou)


################################################################################
# Dataset


class DataSet(object):

    def __init__(self, image_infos=[]):
        self.image_infos = image_infos

    def __repr__(self):
        return f"DataSet(image_infos={self.image_infos})"


def load_dataset(dataset_dir):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    image_infos = []

    extensions = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]

    for ext in extensions:
        for idx, image_file in enumerate(sorted(glob.glob(os.path.join(dataset_dir, f"*.{ext}")))):
            image_stem = image_file[:-len(ext) - 1]
            image_xml = image_stem + ".xml"
            gt_bboxes = []
            if os.path.exists(image_xml):
                gt_bboxes = load_image_metadata_from_xml(image_xml)
            image_infos.append(ImageInfo(image_stem=image_stem, image_file=image_file, gt_bboxes=gt_bboxes))

    return DataSet(image_infos)


################################################################################
# Concurrent


class MviInferenceClient(object):

    def __init__(self, api_url):
        if not api_url:
            raise ValueError("api_url is empty")
        self._api_url = api_url
        m = re.match(r"^.*/([^/]+)$", api_url)
        self.model_id = m.group(1)

    def infer_file(self, filepath):
        image_bytes_reader = io.open(filepath, 'rb')
        files = {'files': ['filename.jpg', image_bytes_reader.read(), 'image/jpeg']}
        r = requests.post(self._api_url, files=files)
        return r

    def infer_bytes(self, filepath):
        image_bytes_reader = io.open(filepath, 'rb')
        files = {'files': ['filename.jpg', image_bytes_reader.read(), 'image/jpeg']}
        r = requests.post(self._api_url, files=files)
        return r


################################################################################
# Accuracy Calcuration
def _calc_AP(precisions: list[float], recalls: list[float]) -> float:

    recalls.append(1)
    precisions.append(0)
    max_precision: float = 0
    left_point: float = 0
    average_precision: float = 0
    for idx in range(len(recalls)):
        p: float = precisions[idx]
        r: float = recalls[idx]
        if p < max_precision:
            # calc integration
            average_precision += (r - left_point) * max_precision

            # reset this range
            max_precision = p
            # print(f"{label},{r},{p},{max_precision},add,{r-left_point},{integration_this}")
            left_point = r
        else:
            # keep scan
            max_precision = p
            # print(f"{label},{r},{p},{max_precision},pass,,")
    return average_precision


class ObjectDetectionStat(object):

    def __init__(self):
        self._gt_bboxes: list = []
        self._pd_bboxes: list = []

        self.num_gt: int = 0
        self.num_pd: int = 0

        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.precision: float = 0
        self.recall: float = 0
        self.fmeasure: float = 0

        self.AP: float = 0

    def to_dict(self) -> dict:
        d: dict = vars(self)
        del d["_gt_bboxes"]
        del d["_pd_bboxes"]
        return d

    def calc_accuracy(self):
        self.num_gt = len(self._gt_bboxes)
        self.num_pd = len(self._pd_bboxes)

        self.tp = len([bbox for bbox in self._pd_bboxes if bbox.iou >= 0.5])
        self.fp = len([bbox for bbox in self._pd_bboxes if bbox.iou < 0.5])
        self.fn = len([bbox for bbox in self._gt_bboxes if bbox.iou < 0.5])

        # stat.precision = stat.tp / (stat.tp + stat.fp)
        # stat.recall = stat.tp / (stat.tp + stat.fn)
        if self.tp == 0:
            self.precision = 0
        elif len(self._pd_bboxes) > 0:
            self.precision = self.tp / len(self._pd_bboxes)
        else:
            self.precision = -1

        if self.tp == 0:
            self.recall = 0
        elif len(self._gt_bboxes) > 0:
            self.recall = self.tp / len(self._gt_bboxes)
        else:
            self.recall = -1

        if self.precision == 0 and self.recall == 0:
            self.fmeasure = 0
        elif (self.precision + self.recall) > 0:
            self.fmeasure = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.fmeasure = -1

        self._calc_AP()

    def _calc_AP(self):
        # Calc Precision Recall curve
        pd_bboxes: list[BoundingBox] = sorted(self._pd_bboxes, reverse=True, key=lambda bbox: bbox.confidence)
        tmp_pd_bboxes: list[BoundingBox] = []
        precisions: list[float] = []
        recalls: list[float] = []
        for pd_bbox in pd_bboxes:
            tmp_pd_bboxes.append(pd_bbox)
            tp: int = len([bbox for bbox in tmp_pd_bboxes if bbox.iou >= 0.5])
            fp: int = len([bbox for bbox in tmp_pd_bboxes if bbox.iou < 0.5])
            if tp == 0:
                precision: float = 0
            else:
                precision: float = tp / (tp + fp)

            if tp == 0:
                recall = 0
            elif len(self._gt_bboxes) > 0:
                recall: float = tp / len(self._gt_bboxes)
            else:
                recall = -1

            precisions.append(precision)
            recalls.append(recall)

        # Calc average precision (AP)
        self.AP = _calc_AP(precisions, recalls)


def calc_accuracy_of_object_detection(dataset) -> list[ObjectDetectionStat, dict]:

    image_info: ImageInfo
    gt_bbox: BoundingBox
    pd_bbox: BoundingBox

    _LOGGER.info("Gathering bounding box groups by label")
    label2stat = defaultdict(ObjectDetectionStat)
    for image_info in dataset.image_infos:
        image_info.analyze_iou()
        for gt_bbox in image_info.gt_bboxes:
            label2stat[gt_bbox.label]._gt_bboxes.append(gt_bbox)
            label2stat["__all__"]._gt_bboxes.append(gt_bbox)
        for pd_bbox in image_info.pd_bboxes:
            label2stat[pd_bbox.label]._pd_bboxes.append(pd_bbox)
            label2stat["__all__"]._pd_bboxes.append(pd_bbox)

    _LOGGER.info("Calculating accuracy groups by label")
    for label, stat in label2stat.items():
        stat.calc_accuracy()

    all_stat = label2stat.pop("__all__")
    all_stat.mAP = statistics.mean([stat.AP for label, stat in label2stat.items()])

    return all_stat, label2stat


################################################################################


class ObjectDetectionStatCalculator(object):

    def __init__(self, mvi_client):
        self.mvi_client = mvi_client

    def infer(self, dataset, update_cache=False, num_of_threads=DEFAULT_NUM_OF_THREADS):

        _LOGGER.info("Start infer with num_of_threads=%s", num_of_threads)

        def sample_func(idx, image_info):
            _LOGGER.info("Infering image (%s/%s", idx, len(dataset.image_infos))
            json_file = image_info.image_stem + "_" + self.mvi_client.model_id + ".cache.json"
            if not update_cache and os.path.exists(json_file):
                result_json = json.load(io.open(json_file, "rt"))
            else:
                _LOGGER.info("Sending inference request for %s", image_info.image_file)
                reqeusts_result = self.mvi_client.infer_file(image_info.image_file)
                result_json = reqeusts_result.json()
                with io.open(json_file, "wt") as f:
                    f.write(reqeusts_result.text)

            pd_bboxes = load_image_metadata_from_json(result_json)
            image_info.pd_bboxes = pd_bboxes
            return image_info

        future_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_of_threads) as executor:
            for idx, image_info in enumerate(dataset.image_infos):
                future = executor.submit(sample_func, idx, image_info)
                future_list.append(future)

        return dataset


################################################################################
# MVI Validator main
def main__deployed_model__object_detection__measure_accuracy(dataset_dir: str, api_url: str, ignore_cache=False, num_of_threads: int = DEFAULT_NUM_OF_THREADS) -> list[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(dataset_dir)
    mvi_client = MviInferenceClient(api_url)

    stat_calculator = ObjectDetectionStatCalculator(mvi_client)
    stat_calculator.infer(dataset, update_cache=ignore_cache, num_of_threads=num_of_threads)

    all_stat, label2stat = calc_accuracy_of_object_detection(dataset)
    # all_stat["model_id"] = mvi_client.model_id

    # data frame for summary
    summary_df: pd.DataFrame = pd.DataFrame([all_stat.to_dict()])

    # data frame for each labels
    label2summary_df = pd.DataFrame()
    for label, stat in label2stat.items():
        d = stat.to_dict()
        d["label"] = label
        label2summary_df = pd.concat([label2summary_df, pd.DataFrame([d])])

    # data frame for all bboxes
    image_info: ImageInfo
    gt_bboxes_df = pd.DataFrame()
    pd_bboxes_df = pd.DataFrame()
    for image_info in dataset.image_infos:
        for bbox in image_info.gt_bboxes:
            gt_bboxes_df = pd.concat([gt_bboxes_df, pd.DataFrame([{
                "file": os.path.basename(image_info.image_file),
                "gt-label": bbox.label,
                "iou": bbox.iou,
                "pd-confidence": 0 if bbox.related_pd_bbox is None else bbox.related_pd_bbox.confidence,
            }])])
        for bbox in image_info.pd_bboxes:
            pd_bboxes_df = pd.concat([pd_bboxes_df, pd.DataFrame([{
                "file": os.path.basename(image_info.image_file),
                "pd-label": bbox.label,
                "iou": bbox.iou,
                "pconfidence": bbox.confidence,
            }])])

    return summary_df, label2summary_df, gt_bboxes_df, pd_bboxes_df


def _sort_dataframe(df, df_name, sortkeys):
    sortkeys2 = []
    for sortkey in sortkeys:
        if sortkey not in df.dtypes:
            _LOGGER.warning("The sortkey==%s is not inclued in data frame==%s. Just ignore it", sortkey, df_name)
        else:
            sortkeys2.append(sortkey)

    return df.sort_values(by=sortkeys2)


def cli_main__deployed_model__object_detection__measure_accuracy( \
    dataset_dir: str, api_url: str, ignore_cache=False, num_of_threads: int = DEFAULT_NUM_OF_THREADS, format=DEFAULT_PRINT_FORMAT, sortkeys=DEFAULT_SORT_KEYS, \
    output_tio=sys.stdout, print_summary=True, print_gt_bboxes=False, print_pd_bboxes=False, print_all=False) -> int:
    if print_all:
        print_summary = True
        print_gt_bboxes = True
        print_pd_bboxes = True

    summary_df, label2summary_df, gt_bboxes_df, pd_bboxes_df = main__deployed_model__object_detection__measure_accuracy(dataset_dir, api_url, ignore_cache, num_of_threads)

    if sortkeys:
        summary_df = _sort_dataframe(summary_df, 'summary_df', sortkeys)
        label2summary_df = _sort_dataframe(label2summary_df, 'label2summary_df', sortkeys)
        gt_bboxes_df = _sort_dataframe(gt_bboxes_df, 'gt_bboxes_df', sortkeys)
        pd_bboxes_df = _sort_dataframe(pd_bboxes_df, 'pd_bboxes_df', sortkeys)

    if format.lower() in ("csv", "tsv"):
        sep: str = "\t" if format.lower() == "tsv" else ","
        if print_summary:
            output_tio.write(os.linesep)
            output_tio.write("# Summary" + os.linesep)
            summary_df.to_csv(output_tio, index=False, sep=sep)
        if print_summary:
            output_tio.write(os.linesep)
            output_tio.write("# Summary of each labels" + os.linesep)
            label2summary_df.to_csv(output_tio, index=False, sep=sep)
        if print_gt_bboxes:
            output_tio.write(os.linesep)
            output_tio.write("# Grand truth boundinx boxes" + os.linesep)
            gt_bboxes_df.to_csv(output_tio, index=False, sep=sep)
        if print_pd_bboxes:
            output_tio.write(os.linesep)
            output_tio.write("# Predicted boundinx boxes" + os.linesep)
            pd_bboxes_df.to_csv(output_tio, index=False, sep=sep)
    elif format.lower() in ("md", "markdown"):
        if print_summary:
            output_tio.write(os.linesep)
            output_tio.write("# Summary" + os.linesep)
            summary_df.to_markdown(output_tio, index=False)
            output_tio.write(os.linesep)
        if print_summary:
            output_tio.write(os.linesep)
            output_tio.write("# Summary of each labels" + os.linesep)
            label2summary_df.to_markdown(output_tio, index=False)
            output_tio.write(os.linesep)
        if print_gt_bboxes:
            output_tio.write(os.linesep)
            output_tio.write("# Grand truth boundinx boxes" + os.linesep)
            gt_bboxes_df.to_markdown(output_tio, index=False)
            output_tio.write(os.linesep)
        if print_pd_bboxes:
            output_tio.write(os.linesep)
            output_tio.write("# Predicted boundinx boxes" + os.linesep)
            pd_bboxes_df.to_markdown(output_tio, index=False)
            output_tio.write(os.linesep)

    else:
        raise ValueError(f"Unknown print format=={format}")

    return 0


################################################################################
# Utilities for CLI


class _VersionAction(argparse.Action):

    def __call__(self, parser: argparse.ArgumentParser, namespace, values, option_string=None):
        print(__version__)
        parser.exit()


def _cli_main(*args: list[str]) -> int:
    if not args:
        args = sys.argv[1:]

    logging.basicConfig(stream=sys.stderr, format=LOG_FORMAT, level=logging.WARNING)

    # "mvi-validator" command
    mvi_validator_argparser: ArgumentParser = argparse.ArgumentParser( \
        prog="mvi-validator", description="MVI Validator", \
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=50, width=320))

    mvi_validator_argparser.add_argument('--version', nargs=0, action=_VersionAction, help="Show program's version number and exit")

    _acceptable_levels = list(logging._nameToLevel.keys())
    _acceptable_levels.remove("NOTSET")

    mvi_validator_argparser.add_argument('--loglevel', dest="log_level", metavar="LEVEL", nargs=None, default=None, help=f"Log level either {_acceptable_levels}.")
    mvi_validator_argparser.set_defaults(handler=mvi_validator_argparser.print_help)
    mvi_validator__argparser__sub_parsers_action: Action = mvi_validator_argparser.add_subparsers(help="sub commands")

    # "mvi-validator deployed-model" command
    mvi_validator__deployed_model__argparser: ArgumentParser = mvi_validator__argparser__sub_parsers_action.add_parser("deployed-model", aliases=["dm"], help="validate object detection")
    mvi_validator_argparser.set_defaults(handler=mvi_validator__deployed_model__argparser.print_help)
    mvi_validator__deployed_model__argparser__sub_parsers_action: Action = mvi_validator__deployed_model__argparser.add_subparsers(help="sub commands")

    # "mvi-validator deployed-model detection" command
    mvi_validator__deployed_model__argparser: ArgumentParser = mvi_validator__deployed_model__argparser__sub_parsers_action.add_parser("detection", aliases=["d"], help="validate object detection")
    mvi_validator__deployed_model__argparser.add_argument(dest="dataset_dir", metavar="DIR", nargs='?', default=None, help="Download directry. Default is ./download")
    mvi_validator__deployed_model__argparser.add_argument("--api", dest="api_url", metavar="URL", nargs=None, default=None, help="API endpoint of deployed model")
    mvi_validator__deployed_model__argparser.add_argument('--summary', "--print-summary", dest="print_summary", action='store_true', default=True, help="Print summary table (default True)")
    mvi_validator__deployed_model__argparser.add_argument('--gt', "--print-gt-bbox", dest="print_gt_bboxes", action='store_true', default=False, help="Print grand truth bounding box table (default False)")
    mvi_validator__deployed_model__argparser.add_argument('--pd', "--print-pd-bbox", dest="print_pd_bboxes", action='store_true', default=False, help="Print predicted   bounding box table (default False)")
    mvi_validator__deployed_model__argparser.add_argument('--all', "--print-all", dest="print_all", action='store_true', default=False, help="Print all tables (default False)")
    mvi_validator__deployed_model__argparser.add_argument("--format", dest="format", metavar="md|markdown|csv|tsv", nargs=None, default=DEFAULT_PRINT_FORMAT, help=f"Print resutls as this format. (default: {DEFAULT_PRINT_FORMAT})")
    mvi_validator__deployed_model__argparser.add_argument("--sortkey", dest="sortkeys", metavar="COLNAME", nargs='*', default=DEFAULT_SORT_KEYS, help=f"Sort the results by these keys. (default: {DEFAULT_SORT_KEYS})")
    mvi_validator__deployed_model__argparser.add_argument('--ignore-cache', dest="ignore_cache", action='store_true', default=False, help="Ignore inference cache with new result")
    mvi_validator__deployed_model__argparser.add_argument("--parallel", dest="num_of_threads", metavar="INT", nargs=None, type=int, default=DEFAULT_NUM_OF_THREADS, help="Run n jobs in parallel")
    mvi_validator__deployed_model__argparser.add_argument('--loglevel', dest="log_level", metavar="LEVEL", nargs=None, default=None, help=f"Log level either {_acceptable_levels}.")
    mvi_validator__deployed_model__argparser.set_defaults(handler=cli_main__deployed_model__object_detection__measure_accuracy)

    arg_ns: Namespace = mvi_validator_argparser.parse_args(args)
    key2value = vars(arg_ns)

    if "version" in key2value:
        del key2value["version"]

    if "log_level" in key2value:
        log_level: str = key2value.pop("log_level")
        if log_level:
            _LOGGER.setLevel(log_level.upper())

    if "handler" in key2value:
        handler = key2value.pop("handler")
        _LOGGER.debug("Found argparser.parse_args: %s", key2value)
        ans: int = handler(**key2value)
        if ans == 0:
            _LOGGER.debug("Cheers!🍺")
    else:
        mvi_validator_argparser.print_help()

    return 1


if __name__ == "__main__":
    # sys.exit(cli_main(*sys.argv[1:]))
    sys.exit(_cli_main("home"))
