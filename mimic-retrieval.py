import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, resnet50
from PIL import Image
import sqlite3
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
import clip
import faiss
import json
from transformers import CLIPProcessor, CLIPModel
import pandas as pd


# 数据集根目录
data_root = "/workspace/data/files/"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(0)
os.environ["OMP_NUM_THREADS"] = "0"
torch.set_num_threads(1)
torch.manual_seed(seed=0)

view_pos = ['AP', 'PA', 'LATERAL']


# 滑动窗口函数
def sliding_window(text, window_size=77, step_size=25):
    tokens = clip.tokenize([text])[0]
    token_length = tokens.size(0)
    windows = []

    for i in range(0, token_length - window_size + 1, step_size):
        window = tokens[i:i+window_size]
        windows.append(window)

    if token_length > window_size:
        windows.append(tokens[-window_size:])

    return windows

# 编码超长文本
def encode_long_text(text, model, device, window_size=77, step_size=25):
    windows = sliding_window(text, window_size, step_size)
    encoded_windows = []

    for window in windows:
        with torch.no_grad():
            input_ids = torch.unsqueeze(window, 0).to(device)
            encoded_window = model.encode_text(input_ids).cpu().numpy()
            encoded_windows.append(encoded_window)

    # 平均池化
    encoded_text = np.mean(encoded_windows, axis=0)

    return encoded_text

# 修改后的 encode_captions 函数
def encode_captions(captions, model, device):
    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        batch_captions = captions[idx:idx+bs]
        batch_encoded_captions = []

        for caption in batch_captions:
            # 尝试分割并编码超长文本
            try:
                encoded_caption = encode_long_text(caption, model, device)
            except RuntimeError as e:
                # 捕获并处理运行时错误
                # print(f"Error encoding caption: {caption}")
                # print(e)
                continue

            batch_encoded_captions.append(encoded_caption)

        encoded_captions.append(np.vstack(batch_encoded_captions))

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions



def encode_images(images, image_path,img_dirs, model, feature_extractor, device):
    image_ids = [i for i in images]

    bs = 64
    image_features = []
    for idx in tqdm(range(0, len(images), bs)):
        image_input = [feature_extractor(Image.open(os.path.join(image_path, img_dirs[i].strip('/'), images[i])))
                       for i in range(idx, min(idx + bs, len(images)))]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    # for idx in tqdm(range(0, len(images), bs)):
    #     image_input = [feature_extractor(Image.open(os.path.join(image_path, i)))
    #                    for i in images[idx:idx + bs]]
    #     with torch.no_grad():
    #         image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features


def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    print("encoded_captions shape:", xb.shape)
    print("encoded_images shape:", xq.shape)
    print("Faiss index dimension:", index.d)


    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)
    print("Index shape:", index.ntotal)  # 输出索引中向量的数量
    print("I shape:", I.shape)  # 输出最近邻索引数组的形状

    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    retrieved_ids = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        good_ids = []
        for idx, nn in enumerate(nns_list):
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            good_ids.append(xb_image_ids[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
        retrieved_ids[image_id] = good_ids
    return retrieved_captions, retrieved_ids

def __get_reports_images( img_positions):
    file_name = 'reports.json'
    caption_file = json.load(open(data_root + file_name, 'r'))
    img_captions = []
    img_files = []
    img_dir=[]
    for file_name, report in caption_file.items():
        k = file_name[-23:-4]
        path = file_name[-27:-24]
        pid, sid = k.split('/')
        try:
            # List all available images in each folder
            file_list = os.listdir(data_root + 'images' + '/' + pid + '/' + sid)
            # Select only images in self.view_pos
            file_list = [f for f in file_list if img_positions[f[:-4]] in view_pos]
            # Make sure there is atleast one image in each folder, and a non-empty findings section in each report
            findings = report.get('FINDINGS:', '')
            if len(file_list) and ('FINDINGS:' in report) and (report['FINDINGS:'] != ''):
                img_files.append( file_list[0])
                img_captions.append(findings)
                img_dir.append('/' + pid + '/' + sid)
        except Exception as e:
            pass
    return img_captions, img_files,img_dir

def __get_view_positions():
    file_name = 'mimic-cxr-2.0.0-metadata.csv'
    txt_file = data_root + file_name
    data = pd.read_csv(txt_file, dtype=object)
    data = data.to_numpy().astype(str)
    return dict(zip(data[:,0].tolist(), data[:,4].tolist())), np.unique(data[:,4]).tolist()

max_len = 1024
vocab_file = 'mimic_unigram_1000.model'
vocab = spm.SentencePieceProcessor(model_file=data_root + vocab_file)


img_positions, list_positions = __get_view_positions()
captions , images ,img_dir=  __get_reports_images( img_positions)



device = "cuda" if torch.cuda.is_available() else "cpu"
print('CUDA current_device: {}'.format(torch.cuda.current_device()))
torch.cuda.empty_cache()  # 清理未使用的变量和缓存

clip_model, feature_extractor = clip.load("ViT-B/32", device=device)
print('Encoding captions')

encoded_captions = encode_captions(captions, clip_model, device)

print('Encoding images')
image_ids, encoded_images = encode_images(images, data_root + 'images',img_dir, clip_model, feature_extractor, device)

print('Retrieving neighbors')
index, nns = get_nns(encoded_captions, encoded_images)
retrieved_caps, retrieved_ids = filter_nns(nns, images,captions, image_ids)

# # encoded_captions = encode_captions(captions, clip_model, device, vocab, max_len)
# encoded_captions = encode_captions(captions, clip_model, device)
#
# print('Encoding images')
# image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)
#
# print('Retrieving neighbors')
# index, nns = get_nns(encoded_captions, encoded_images)
# retrieved_caps, retrieved_ids = filter_nns(nns, images, captions, image_ids)

output_file_path = '/workspace/data/files/retrieved_captions.json'
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(retrieved_caps, json_file, ensure_ascii=False, indent=4)

print(f"Retrieved captions saved to {output_file_path}")

output_ids_path = '/workspace/data/files/retrieved_ids.json'
with open(output_ids_path, 'w', encoding='utf-8') as json_file:
    json.dump(retrieved_ids, json_file, ensure_ascii=False, indent=4)

print(f"Retrieved ids saved to {output_ids_path}")
