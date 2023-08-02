import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List
import sys
# sys.path.append("/ssd/lizhaoyang/code/neurips22-cellseg_saltfish/code_saltfish")
# from constants import SAMPLE_SHAPE


def read_json(fpath: Path) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data 
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

class DetectionWriter:
    """This class writes the cell predictions to the designated 
    json file path with the Multiple Point format required by 
    Grand Challenge

    Parameters
    ----------
    output: Path
        path to json output file to be generated
    """

    def __init__(self, output_path: Path):

        if output_path.suffix != '.json':
            output_path = output_path / '.json' 

        self._output_path = output_path
        self._data = {
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        } 

    def add_point(
            self, 
            x: int, 
            y: int,
            class_id: int,
            prob: float, 
            sample_id: int
        ):
        """Recording a single point/cell

        Parameters
        ----------
        x: int
            Cell's x-coordinate in the cell patch
        y: int
            Cell's y-coordinate in the cell patch
        class_id: int
            Class identifier of the cell, either 1 (BC) or 2 (TC)
        prob: float
            Confidence score
        sample_id: str
            Identifier of the sample
        """
        point = {
            "name": "image_{}".format(str(sample_id)),
            "point": [int(x), int(y), int(class_id)],
            "probability": prob}
        
        self._data["points"].append(point)
        
    def add_points(self, points: List, sample_id: str):
        """Recording a list of points/cells

        Parameters
        ----------
        points: List
            List of points, each point consisting of (x, y, class_id, prob)
        sample_id: str
            Identifier of the sample
        """
        # count=0
        # print("加入{}, 长度{}".format(sample_id, len(points)))
        for x, y, c, prob in points:
            self.add_point(x, y, c, prob, sample_id)
            # count += 1
        # print("cell count:",count)

    def save(self):
        """This method exports the predictions in Multiple Point json
        format at the designated path. 
        
        - NOTE: that this will fail if not cells are predicted
        """
        assert len(self._data["points"]) > 0, "No cells were predicted"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
        print(f"Predictions were saved at `{self._output_path}`")

