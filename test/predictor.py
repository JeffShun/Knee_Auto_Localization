from calendar import c
from dis import dis
from typing import IO, Dict
import numpy as np
import torch
import yaml
from skimage.morphology import dilation
from train.config.model_config import network_cfg
import tensorrt as trt
import onnxruntime as ort
import pycuda.driver as pdd
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class PredictConfig:
    def __init__(self, test_cfg):
        # 配置文件
        self.patch_size = test_cfg.get('patch_size')

    def __repr__(self) -> str:
        return str(self.__dict__)


class Predictor:
    def __init__(self, device, model_f, config_f):
        self.device = torch.device(device)
        self.ort_flag = False
        self.tensorrt_flag = False 

        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg

        with open(self.config_f, 'r') as config_f:
            self.test_cfg = PredictConfig(yaml.safe_load(config_f))
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model_f, str):
            # 根据后缀判断类型
            if self.model_f.endswith('.pth'):
                self.load_model_pth()
            elif self.model_f.endswith('.pt'):
                self.load_model_jit()
            elif self.model_f.endswith('.onnx'):
                self.ort_flag = True
                self.load_model_onnx()
            elif self.model_f.endswith('.engine'):
                self.tensorrt_flag = True
                self.load_model_engine()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device)

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def load_model_onnx(self) -> None:
        # 加载onnx
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.ort_session = ort.InferenceSession(self.model_f, session_options=session_options, providers=['CUDAExecutionProvider'])

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model_f, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = pdd.Stream()
        for i, binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = pdd.pagelocked_empty(size, dtype)
            device_mem = pdd.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_inference(self, context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [pdd.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [pdd.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def max_index(self, img):
        b, c, d, h, w = img.shape
        _, max_indices = torch.max(img.view(b, c, -1), dim=-1)
        max_indices_3d = np.array([np.unravel_index(idx, (d, h, w)) for idx in max_indices.view(-1).tolist()])
        max_indices_3d = torch.from_numpy(max_indices_3d.reshape(b, c, 3)).to(img.device)
        return max_indices_3d
    
    def predict(self, volume: np.ndarray):
        # 数据预处理
        shape = volume.shape
        volume = torch.from_numpy(volume).float()[None, None]
        volume = self._normlize(volume)
        # 模型预测
        kp_heatmap, regression = self._forward(volume)
        max_index = self.max_index(kp_heatmap).squeeze().cpu().detach().numpy()
        shift = regression.squeeze().cpu().detach().numpy()
        kp_mask = np.zeros(kp_heatmap.shape[1:], dtype="uint8")
        for i in range(max_index.shape[0]):
            ori_kp_coords = max_index[i]
            print(shift[i, ori_kp_coords[0], ori_kp_coords[1], ori_kp_coords[2]])
            final_kp_coords = ori_kp_coords + shift[i, ori_kp_coords[0], ori_kp_coords[1], ori_kp_coords[2]]
            kp_mask[i][tuple(np.round(final_kp_coords, decimals=0).astype("int32"))] = 1

        # 对1，2，3点的z坐标进行矫正，使其相等
        points = []
        for i in range(kp_mask.shape[0]):
            kp_i = list(zip(*np.where(kp_mask[i] == 1)))[0]
            points.append(kp_i)
        points = np.array(points)
        z_correct = int((points[0,0] + points[1,0] + points[2,0]) / 3 + 0.5)   
        points[0,0] = z_correct
        points[1,0] = z_correct
        points[2,0] = z_correct
        out_mask = np.zeros(shape, dtype="uint8")
        for i in range(kp_mask.shape[0]):
            mask_i = np.zeros(shape, dtype="uint8")
            mask_i[tuple(points[i])] = 1
            mask_dilate = dilation(mask_i, np.ones([2, 4, 4]))
            out_mask[mask_dilate==1] = i+1
        return out_mask, kp_heatmap.squeeze().cpu().detach().numpy(), points
    
    def _forward(self, volume: torch.Tensor):
        # tensorrt预测
        if self.tensorrt_flag:
            cuda_ctx = pycuda.autoinit.context
            cuda_ctx.push()
            # 动态输入
            volume = np.ascontiguousarray(volume.numpy())
            self.context.active_optimization_profile = 0
            origin_inputshape = self.context.get_binding_shape(0)
            origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3], origin_inputshape[4] = volume.shape
            # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size
            self.context.set_binding_shape(0, (origin_inputshape))  
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, self.context)
            inputs[0].host = volume
            trt_outputs = self.trt_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream, batch_size=1)
            if cuda_ctx:
                cuda_ctx.pop()
            shape_of_output1 = [1, 5, 96, 192, 192]
            shape_of_output2 = [1, 5, 96, 192, 192, 3]
            output1 = trt_outputs[0].reshape(shape_of_output1)
            output2 = trt_outputs[1].reshape(shape_of_output2)
            output = torch.from_numpy(output1),  torch.from_numpy(output2)

        elif self.ort_flag:
            ort_outputs = self.ort_session.run(None, {"input": volume.numpy().astype(np.float32)})   
            shape_of_output1 = [1, 5, 96, 192, 192]
            shape_of_output2 = [1, 5, 96, 192, 192, 3]
            output1 = ort_outputs[0].reshape(shape_of_output1)
            output2 = ort_outputs[1].reshape(shape_of_output2)
            output = torch.from_numpy(output1), torch.from_numpy(output2)
        else:
            # pytorch预测
            with torch.no_grad():
                patch_gpu = volume.to(self.device)
                output = self.net(patch_gpu)
        return output

    def _normlize(self, data, win_clip=None):
        if win_clip is not None:
            data = torch.clip(data, win_clip[0], win_clip[1])
        data = (data - data.min())/(data.max() - data.min())
        return data 

    def _resize_torch(self, data, scale, mode="trilinear"):
        return torch.nn.functional.interpolate(data, size=scale, mode=mode)    

