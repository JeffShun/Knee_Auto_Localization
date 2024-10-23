import os

model = "ResUNET+Refine+GaussianMSE"
test_epochs = range(100,200,2)
log_path = "./data/output/%s/test_epochs.log"%(model)
directory = os.path.dirname(log_path)
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

for i in test_epochs:
    print("Begin test epoch %d !"%(i))
    f = open(log_path, 'a+')  
    print("Begin test epoch %d !"%(i), file=f)
    f.close()
    os.system("python main.py --output_path data/output/%s --model_file ../train/checkpoints/%s/%d.pth"%(model, model, i))
    os.system("python analysis_tools/cal_metrics.py --data_path ./data/output/%s/data_for_metrics --metrics_path ./data/output/%s/epoch-%d.csv --print_path %s"%(model, model, i, log_path))

os.system("python analysis_tools/parse_txt_result.py --src_path %s --dst_path %s"%(log_path, log_path.replace(".log", ".csv")))

