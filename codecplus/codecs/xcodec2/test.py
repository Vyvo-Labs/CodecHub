import os
import torch
import soundfile as sf
from  modeling_xcodec2 import XCodec2Model
from tqdm import tqdm

# 设置模型路径
model_path = "HKUST-Audio/xcodec2"

# 加载预训练的 XCodec2 模型
Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()

model = Codec_model

# 设置 LibriSpeech test-clean 数据集的路径
input_dir = "/aifs4su/data/zheny/bigcodec_final/v10_31_final_ml_semantic_real_baodi_perception_loss/eval_tools/test_gt"  # 请替换为实际的 test-clean 数据集路径

# 设置重建后音频文件的输出文件夹
output_dir = "/aifs4su/data/zheny/opensource/recon_clean_100_debug"  # 请替换为你希望保存重建音频的文件夹路径

# 如果输出文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 遍历输入文件夹中的所有子文件夹和音频文件
for root, dirs, files in os.walk(input_dir):
    # 计算相对于 input_dir 的相对路径，以保持目录结构
    rel_path = os.path.relpath(root, input_dir)
    
    # 创建对应的输出子文件夹
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 使用 tqdm 进行进度显示
    for file in tqdm(files, desc=f"Processing {rel_path}"):
        if file.endswith(".wav") or file.endswith(".flac"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subdir, file)
            # try:
                # 读取音频文件
            wav, sr = sf.read(input_path)
            
            # 将音频数据转换为张量，并移动到 GPU
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) 
            
            with torch.no_grad():
                # 编码音频
                vq_code = model.encode_code(input_waveform=wav_tensor)
                
                # 解码以重建音频
                recon_wav = model.decode_code(vq_code).cpu()
            
            # 假设重建的 wav 形状为 [batch, channels, samples]
            # 将重建后的音频保存到输出路径
            sf.write(output_path, recon_wav[0, 0, :].numpy(), sr)
            
            # except Exception as e:
            # print(f"处理文件 {input_path} 时出错: {e}")

print("完成！请检查重建后的音频文件位于：", output_dir)
