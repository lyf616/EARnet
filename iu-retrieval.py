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


# 数据集根目录
data_root = "/workspace/nlp-cgi/Datasets/NLMCXR/"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(0)
os.environ["OMP_NUM_THREADS"] = "0"
torch.set_num_threads(1)
torch.manual_seed(seed=0)


#
# import jieba
# jieba.load_userdict('/workspace/cgi/Datasets/ours/dict')

# def encode_captions(captions, model, device):
#     bs = 256
#     encoded_captions = []
#
#     for idx in tqdm(range(0, len(captions), bs)):
#         with torch.no_grad():
#             # Extract the current batch of captions
#             batch_captions = captions[idx:idx + bs]
#
#             # Tokenize Chinese captions using jieba
#             batch_tokenized_captions = [" ".join(jieba.lcut(caption)) for caption in batch_captions]
#
#             # Convert tokenized captions to input_ids
#             input_ids = clip.tokenize(batch_tokenized_captions).to(device)
#
#             # Encode captions and append to the list
#             encoded_captions.append(model.encode_text(input_ids).cpu().numpy())
#
#     encoded_captions = np.concatenate(encoded_captions)
#
#     return encoded_captions

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
                print(f"Error encoding caption: {caption}")
                print(e)
                continue

            batch_encoded_captions.append(encoded_caption)

        encoded_captions.append(np.vstack(batch_encoded_captions))

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

# def encode_captions(captions, model, device, vocab, max_len):
#     bs = 256
#     encoded_captions = []
#
#     for idx in tqdm(range(0, len(captions), bs)):
#         with torch.no_grad():
#             # Extract the current batch of captions
#             batch_captions = captions[idx:idx + bs]
#
#             # Placeholder for encoded captions in the current batch
#             batch_encoded_captions = []
#
#             # Encode each caption in the batch
#             for caption in batch_captions:
#                 # Encode caption using vocab
#                 encoded_caption = [vocab.bos_id()] + vocab.encode(caption) + [vocab.eos_id()]
#
#                 # Pad the encoded caption to max_len
#                 padded_caption = np.ones(max_len, dtype=np.float32) * vocab.pad_id()
#                 padded_caption[:min(len(encoded_caption), max_len)] = encoded_caption[
#                                                                       :min(len(encoded_caption), max_len)]
#
#                 # Append the padded caption to the batch
#                 batch_encoded_captions.append(padded_caption)
#
#             batch_encoded_captions = torch.tensor(batch_encoded_captions).to(device)
#
#             # Append the batch of encoded captions to the overall list
#             encoded_captions.append(batch_encoded_captions.long().cpu().numpy())
#
#     encoded_captions = np.concatenate(encoded_captions)
#
#     return encoded_captions


def encode_images(images, image_path, model, feature_extractor, device):
    image_ids = [i for i in images]

    bs = 64
    image_features = []

    for idx in tqdm(range(0, len(images), bs)):
        image_input = [feature_extractor(Image.open(os.path.join(image_path, i)))
                       for i in images[idx:idx + bs]]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

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

#
# def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
#     """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
#     retrieved_captions = {}
#     retrieved_ids = {}
#     for nns_list, image_id in zip(nns, xq_image_ids):
#         good_nns = []
#         good_ids = []
#         for idx, nn in enumerate(nns_list):
#             if xb_image_ids[nn] == image_id:
#                 continue
#             good_nns.append(captions[nn])
#             good_ids.append(xb_image_ids[nn])
#             if len(good_nns) == 7:
#                 break
#         assert len(good_nns) == 7
#         retrieved_captions[image_id] = good_nns
#         retrieved_ids[image_id] = good_ids
#     return retrieved_captions, retrieved_ids


def filter_nns(nns, xb_image_ids, captions, encoded_images,xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    retrieved_ids = {}
    retrieved_features = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        good_ids = []
        good_features = []
        for idx, nn in enumerate(nns_list):
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            good_ids.append(xb_image_ids[nn])
            good_features.append(encoded_images[nn].tolist())
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
        retrieved_ids[image_id] = good_ids
        retrieved_features[image_id] = good_features

    return retrieved_captions, retrieved_ids,retrieved_features


#
# # 图像预处理转换
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


max_len = 1024
vocab_file = 'nlmcxr_unigram_1000.model'
vocab = spm.SentencePieceProcessor(model_file=data_root + vocab_file)
captions = []
images = []

with open(data_root + 'file2label.json') as f:
    file_labels = json.load(f)
with open(data_root + 'reports_ori.json') as f:
    reports = json.load(f)
file_list = [k for k in reports.keys()]
file_report = reports
filtered_file_report = {}
for k, v in file_report.items():
    if (len(v['image']) > 0) and (('FINDINGS' in v['report']) and (v['report'][
                                                                       'FINDINGS'] != '')):  # or (('IMPRESSION' in v['report']) and (v['report']['IMPRESSION'] != ''))):
        filtered_file_report[k] = v
    file_report = filtered_file_report
    file_list = [k for k in file_report.keys()]
# 遍历文件夹，读取图像和文本数据
for filename in tqdm(file_list):
    image_id = file_report[filename]["image"][0] + '.png'

    label = file_labels[filename]

    report_text = file_report[filename]["report"]["FINDINGS"]

    captions.append(report_text)
    images.append(image_id)

image_path = os.path.join(data_root, "images")

device = "cuda" if torch.cuda.is_available() else "cpu"
print('CUDA current_device: {}'.format(torch.cuda.current_device()))
torch.cuda.empty_cache()  # 清理未使用的变量和缓存

clip_model, feature_extractor = clip.load("ViT-B/32", device=device)
print('Encoding captions')

# encoded_captions = encode_captions(captions, clip_model, device, vocab, max_len)
encoded_captions = encode_captions(captions, clip_model, device)

print('Encoding images')
image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)

print('Retrieving neighbors')
index, nns = get_nns(encoded_captions, encoded_images)
retrieved_caps, retrieved_ids, retrieved_features = filter_nns(nns, images, captions,encoded_images, image_ids)

output_file_path = '/workspace/nlp-cgi/Datasets/NLMCXR/retrieved_captions.json'
with open(output_file_path, 'w', encoding='gb2312') as json_file:
    json.dump(retrieved_caps, json_file, ensure_ascii=False, indent=4)

print(f"Retrieved captions saved to {output_file_path}")

output_ids_path = '/workspace/nlp-cgi/Datasets/NLMCXR/retrieved_ids.json'
with open(output_ids_path, 'w', encoding='gb2312') as json_file:
    json.dump(retrieved_ids, json_file, ensure_ascii=False, indent=4)

print(f"Retrieved ids saved to {output_ids_path}")


output_features_path = '/workspace/nlp-cgi/Datasets/NLMCXR/retrieved_features.json'
with open(output_features_path, 'w', encoding='gb2312') as json_file:
    json.dump(retrieved_features, json_file, ensure_ascii=False, indent=4)

print(f"Retrieved features saved to {output_features_path}")
