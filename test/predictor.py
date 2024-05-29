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
        self.net.to(self.device).half()

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device).half()

    def load_model_onnx(self) -> None:
        # 加载onnx
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.ort_session = ort.InferenceSession(self.model_f, session_options=session_options, providers=['CUDAExecutionProvider'])

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model.model_f, 'rb') as f:
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


    def predict(self, volume: np.ndarray):
        # 数据预处理
        shape = volume.shape
        volume = torch.from_numpy(volume).float()[None, None]
        volume = self._normlize(volume)
        volume = self._resize_torch(volume, self.test_cfg.patch_size)
        # 模型预测
        kp_heatmap  = self._forward(volume)
        # 数据后处理
        kp_heatmap = self._resize_torch(kp_heatmap, shape)
        kp_heatmap = kp_heatmap.squeeze().cpu().detach().numpy()
        ori_shape = kp_heatmap.shape
        C = ori_shape[0]
        kp_arr = kp_heatmap.reshape(C,-1)
        max_index = np.argmax(kp_arr,1)
        max_p = (np.arange(C), max_index)
        kp_mask = np.zeros_like(kp_arr, dtype="uint8")
        kp_mask[max_p] = 1
        kp_mask = kp_mask.reshape(ori_shape)
        out_mask = np.zeros(shape, dtype="uint8")
        for i in range(kp_mask.shape[0]):
            kp_dilate = dilation(kp_mask[i], np.ones([2, 4, 4]))
            out_mask[kp_dilate==1] = i+1
        return out_mask, kp_heatmap

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
            shape_of_output = [1, 5, 96, 192, 192]
            output = trt_outputs[1].reshape(shape_of_output)
            output = torch.from_numpy(output)

        elif self.ort_flag:
            ort_outputs = self.ort_session.run(None, {"input": volume.numpy().astype(np.float32)})   
            shape_of_output = [1, 5, 96, 192, 192]
            output = ort_outputs[1].reshape(shape_of_output)
            output = torch.from_numpy(output)
        else:
            # pytorch预测
            with torch.no_grad():
                patch_gpu = volume.half().to(self.device)
                _, output = self.net(patch_gpu)
        return output

    def _normlize(self, data, win_clip=None):
        if win_clip is not None:
            data = torch.clip(data, win_clip[0], win_clip[1])
        data = (data - data.min())/(data.max() - data.min())
        return data 

    def _resize_torch(self, data, scale, mode="trilinear"):
        return torch.nn.functional.interpolate(data, size=scale, mode=mode)    

