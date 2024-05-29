
import onnx
import torch
import argparse
import torch.nn
import os, sys
import tensorrt as trt
from onnxsim import simplify
import netron
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from config.model_config import network_cfg


# D:/软件包/AI定位软件包/TensorRT-8.4.1.5/bin/trtexec.exe --onnx=./checkpoints/onnx_model/Conditional-Restormer-newDC-twoCNN-SSIM-L1-TV.onnx --saveEngine=./checkpoints/trt_model/model.engine --minShapes=images:1x4x128x128 --optShapes=images:6x3x512x512 --maxShapes=images:12x3x1024x1024 --useCudaGraph --noTF32 --fp16
# D:/软件包/AI定位软件包/TensorRT-8.4.1.5/bin/trtexec.exe --onnx=./checkpoints/onnx_model/model.onnx --saveEngine=./checkpoints/trt_model/model.engine --fp16 --workspace=512 --dumpProfile --dumpLayerInfo --profilingVerbosity=detailed --verbose
def load_model(model_path):
    model = network_cfg.gen_network
    checkpoint = torch.load(model_path, map_location={"cuda:0":"cuda:0","cuda:1":"cuda:0","cuda:2":"cuda:0","cuda:3":"cuda:0"})
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/BigData/Conditional-ResUnet-DPSR-twoCNN-SSIM-L1-TV/Generator/3.pth')
    parser.add_argument('--output_onnx', type=str, default='./checkpoints/onnx_model')
    parser.add_argument('--output_trt', type=str, default='./checkpoints/trt_model')
    args = parser.parse_args()
    return args


def torch2onnx(model_path, onnx_path):
    model = load_model(model_path)
    model.eval()
    # 定义示例输入
    dummy_input = torch.ones((1, 4, 256, 256),requires_grad=True).cuda()
    # Export the model   
    torch.onnx.export(
        model,                                                 # model being run 
        dummy_input,                                           # model input (or a tuple for multiple inputs) 
        onnx_path,                                             # where to save the model  
        export_params=True,                                    # store the trained parameter weights inside the model file 
        opset_version=11,                                      # the ONNX version to export the model to 
        do_constant_folding=True,                              # whether to execute constant folding for optimization 
        input_names = ['input'],                               # the model's input names 
        output_names = ['output'],                             # the model's output names 
        dynamic_axes = {"input":[0,2,3]}                       # dynamic axes
        )
    print('Model has been converted to ONNX!') 

    # 下面代码主要是为了防止模型过大时显存不足无法生成engine
    torch.cuda.empty_cache()
    onnx_model = onnx.load(onnx_path)  # load onnx model
    model_simp, check = simplify(onnx_model, skipped_optimizers=['extract_constant_to_initializer'])
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print('Onnx model has been succesfully simplified!')
    # netron.start(onnx_path)

 
def onnx2trt(onnx_file_path, engine_file_path):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 1、动态输入第一点必须要写的
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 1  # trt推理时最大支持的batchsize
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = 16*(1 << 32) 
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path')
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # 重点: 动态输入时候需要 分别为最小输入、常规输入、最大输入
        profile = builder.create_optimization_profile() 
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        profile.set_shape("input", (1, 4, 128, 128), (1, 4, 512, 512), (1, 4, 1024, 1024))
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    output_onnx_dir = args.output_onnx
    output_trt_dir = args.output_trt
    print("Beigin torch2onnx!")
    os.makedirs(output_onnx_dir, exist_ok=True)
    onnx_path = os.path.join(output_onnx_dir, 'Conditional-ResUnet-newDC-twoCNN-SSIM-L1-TV.onnx')
    torch2onnx(model_path, onnx_path)

    print("Beigin onnx2trt!")
    os.makedirs(output_trt_dir, exist_ok=True)
    engine_file_path = os.path.join(output_trt_dir, 'model.engine')
    onnx2trt(onnx_path, engine_file_path)


