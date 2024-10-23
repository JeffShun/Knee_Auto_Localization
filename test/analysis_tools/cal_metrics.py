"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
import os
import numpy as np
from runstats import Statistics
import pandas as pd

class ERROR_FUNCS():
    def __init__(self, datas):
        self.dist_threshold = datas["dist_threshold"]
        self.angle_threshold = datas["angle_threshold"]
        self.pred = datas["pred"]
        self.label = datas["label"]
        self.imgsize = datas["imgsize"]
        
        self.abs_error, self.ret_error = self.cal_shift_errors()
        self.sagittal_angle_error, self.axial_angle_error, self.coronal_angle_error = self.cal_angle_errors()


    @staticmethod
    def cal_normal_vec(points):
        point1, point2, point3, point4, point5 = points
        # 计算冠状面的法线向量
        v1 = (0, 0, 1)  # 物理横断面法向量
        v2 = point1 - point2 if point1[0]>point2[0] else point2 - point1
        v2 /= (np.linalg.norm(v2) + 1e-8)
        normal_coronal = tuple(np.cross(v1, v2))
        normal_coronal /= (np.linalg.norm(normal_coronal) + 1e-8)
        # 计算横断面的法线向量
        v1 = point4 - point5 if point4[0]>point5[0] else point5 - point4
        v1 /= (np.linalg.norm(v1) + 1e-8)
        v2 = normal_coronal  # 冠状面法向量
        normal_axial = np.cross(v1, v2)
        normal_axial /= (np.linalg.norm(normal_axial) + 1e-8)
        # 计算矢状面的法线向量
        normal_sagittal = point1 - point2 if point1[0]>point2[0] else point2 - point1
        normal_sagittal /= (np.linalg.norm(normal_sagittal) + 1e-8)
        return normal_sagittal, normal_axial, normal_coronal
    
    def cal_shift_errors(self):
        abs_error = np.sqrt(np.sum((self.pred - self.label)**2, 1))
        ret_error = abs_error / np.sqrt(np.sum(self.imgsize**2))
        return abs_error, ret_error
    
    def cal_angle_errors(self):
        normal_sagittal_pred, normal_axial_pred, normal_coronal_pred = self.cal_normal_vec(self.pred)
        normal_sagittal_label, normal_axial_label, normal_coronal_label = self.cal_normal_vec(self.label)
        iproduct_sagittal = np.dot(normal_sagittal_pred, normal_sagittal_label)
        sagittal_angle_error = 180*np.arccos(iproduct_sagittal)/np.pi
        iproduct_axial = np.dot(normal_axial_pred, normal_axial_label)
        axial_angle_error = 180*np.arccos(iproduct_axial)/np.pi
        iproduct_coronal = np.dot(normal_coronal_pred, normal_coronal_label)
        coronal_angle_error = 180*np.arccos(iproduct_coronal)/np.pi
        return sagittal_angle_error, axial_angle_error, coronal_angle_error

    def accuracy(self):
        if np.all(self.abs_error < self.dist_threshold) \
            and self.sagittal_angle_error < self.angle_threshold \
            and self.axial_angle_error < self.angle_threshold \
            and self.coronal_angle_error < self.angle_threshold:
            return 1
        else:
            return 0
        
    def abs_error1(self):
        return self.abs_error[0]

    def abs_error2(self):
        return self.abs_error[1]

    def abs_error3(self):
        return self.abs_error[2]

    def abs_error4(self):
        return self.abs_error[3]

    def abs_error5(self):
        return self.abs_error[4]

    def ret_error1(self):
        return self.ret_error[0]

    def ret_error2(self):
        return self.ret_error[1]

    def ret_error3(self):
        return self.ret_error[2]

    def ret_error4(self):
        return self.ret_error[3]

    def ret_error5(self):
        return self.ret_error[4]

    def sag_error(self):
        return self.sagittal_angle_error
    
    def axi_error(self):
        return self.axial_angle_error
        
    def cor_error(self):
        return self.coronal_angle_error


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metric_funcs = metric_funcs
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.metrics_data = {metric:[] for metric in metric_funcs}

    def push(self, push_data):
        error_funcs = ERROR_FUNCS(push_data)
        for metric, func in self.metric_funcs.items():
            val = val = func(error_funcs)
            self.metrics[metric].push(val)
            self.metrics_data[metric].append((push_data["pid"], val))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}
    

    def save(self, save_path):
        df = pd.DataFrame()
        for method, values in self.metrics_data.items():
            labels, scores = zip(*values)
            df[method] = scores
        df['pid'] = labels
        df = df[['pid'] + [col for col in df.columns if col != 'pid']]
        df.to_csv(save_path, index=False)


    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = list(means)
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args):
    dist_threshold = args.dist_threshold
    angle_threshold = args.angle_threshold

    METRIC_FUNCS = {
        "ACCURACY": lambda ef: ef.accuracy(),
        "POINT_ERROR1": lambda ef: ef.abs_error1(),
        "POINT_ERROR2": lambda ef: ef.abs_error2(),
        "POINT_ERROR3": lambda ef: ef.abs_error3(),
        "POINT_ERROR4": lambda ef: ef.abs_error4(),
        "POINT_ERROR5": lambda ef: ef.abs_error5(),
        "RET_ERROR1": lambda ef: ef.ret_error1(),
        "RET_ERROR2": lambda ef: ef.ret_error2(),
        "RET_ERROR3": lambda ef: ef.ret_error3(),
        "RET_ERROR4": lambda ef: ef.ret_error4(),
        "RET_ERROR5": lambda ef: ef.ret_error5(),
        "SAG_ERROR": lambda ef: ef.sag_error(),
        "AXI_ERROR": lambda ef: ef.axi_error(),
        "COR_ERROR": lambda ef: ef.cor_error()
        }
    metrics = Metrics(METRIC_FUNCS)

    for sample in args.data_path.iterdir():
        if not str(sample).endswith("npz"):
            continue
        pid = str(sample).replace("\\","/").split("/")[-1].replace("npz","")
        data = np.load(sample, allow_pickle=True)
        pred = data['pred']
        label = data['label']
        imgsize = data['imgsize']
        push_data = {
            "pid": pid,
            "imgsize": imgsize,
            "pred": pred,
            "label": label,
            "dist_threshold": dist_threshold,
            "angle_threshold": angle_threshold
        }

        metrics.push(push_data)

    return metrics


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",
                        default=pathlib.Path("./data/output/newdata/ResUNET+Refine+AUG+DirectionLoss/data_for_metrics"),
                        type=pathlib.Path)
    parser.add_argument("--metrics_path",
                        default=pathlib.Path("./data/output/newdata/ResUNET+Refine+AUG+DirectionLoss/data_for_metrics/metrics.csv"),
                        type=pathlib.Path)
    parser.add_argument('--dist_threshold',
                        default=10.0,
                        type=float)
    parser.add_argument('--angle_threshold',
                        default=3.0,
                        type=float)
    parser.add_argument('--print_path',
                        default='',
                        type=str)
    args = parser.parse_args()
    metrics = evaluate(args)
    metrics.save(args.metrics_path)
    print(metrics)
    if os.path.exists(args.print_path):
        f = open(args.print_path, 'a+')  
        print(metrics, file=f)
        f.close()
