import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from sklearn.cluster import KMeans
import sqlite3
import numpy as np
from scipy.spatial.distance import cosine


# cmn+retrieval


# --- Transformer Modules ---
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1, 0, 2)  # (V,B,E)
        query = query.permute(1, 0, 2)  # (Q,B,E)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask,
                                    attn_mask=att_mask)  # (Q,B,E), (B,Q,V)

        embed = self.normalize(embed + query)  # (Q,B,E)
        embed = embed.permute(1, 0, 2)  # (B,Q,E)
        return embed, att  # (B,Q,E), (B,Q,V)


class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input)  # (B,L,E)
        output = self.normalize(output + input)  # (B,L,E)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, pad_mask=None, att_mask=None):
        emb, att = self.attention(input, input, pad_mask, att_mask)
        emb = self.fwd_layer(emb)
        return emb, att


class TransformerLayer2(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        emb, att = self.attention(input, query, pad_mask, att_mask)
        emb = self.fwd_layer(emb)
        return emb, att


# CMN Transformer
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)

        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1))
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
        embeddings = embeddings + responses
        # Memory querying and responding for textual features

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                 num_tokens=1, num_posits=1, token_embedding=None, posit_embedding=None):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim) if not token_embedding else token_embedding
        self.posit_embedding = nn.Embedding(num_posits, embed_dim) if not posit_embedding else posit_embedding
        self.transform = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_index=None, token_embed=None, pad_mask=None, pad_id=-1, att_mask=None):
        if token_index != None:
            if pad_mask == None:
                pad_mask = (token_index == pad_id)  # (B,L)
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)  # (B,L)
            posit_embed = self.posit_embedding(posit_index)  # (B,L,E)
            token_embed = self.token_embedding(token_index)  # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed)  # (B,L,E)
        elif token_embed != None:
            posit_index = torch.arange(token_embed.shape[1]).unsqueeze(0).repeat(token_embed.shape[0], 1).to(
                token_embed.device)  # (B,L)
            posit_embed = self.posit_embedding(posit_index)  # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed)  # (B,L,E)
        else:
            raise ValueError('token_index or token_embed must not be None')

        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]

        return final_embed  # (B,L,E)


# --- Convolution Modules ---
class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        if 'res' in model_type.lower():  # resnet, resnet-50, resnest-50, ...
            modules = list(model.children())[:-1]  # Drop the FC layer
            self.feature = nn.Sequential(*modules[:-1])
            self.average = modules[-1]
        elif 'dense' in model_type.lower():  # densenet, densenet-121, densenet121, ...
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')

    def forward(self, input):
        wxh_features = self.feature(input)  # (B,2048,W,H)
        avg_features = self.average(wxh_features)  # (B,2048,1,1)
        avg_features = avg_features.view(avg_features.shape[0], -1)  # (B,2048)
        return avg_features, wxh_features


class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        img = input[0]  # (B,V,C,W,H)
        pos = input[1]  # (B,V)
        B, V, C, W, H = img.shape

        img = img.view(B * V, C, W, H)
        avg, wxh = self.model(img)  # (B*V,F), (B*V,F,W,H)
        avg = avg.view(B, V, -1)  # (B,V,F)
        wxh = wxh.view(B, V, wxh.shape[-3], wxh.shape[-2], wxh.shape[-1])  # (B,V,F,W,H)

        msk = (pos == -1)  # (B,V)
        msk_wxh = msk.view(B, V, 1, 1, 1).float()  # (B,V,1,1,1) * (B,V,F,C,W,H)
        msk_avg = msk.view(B, V, 1).float()  # (B,V,1) * (B,V,F)
        wxh = msk_wxh * (-1) + (1 - msk_wxh) * wxh
        avg = msk_avg * (-1) + (1 - msk_avg) * avg

        wxh_features = wxh.max(dim=1)[0]  # (B,F,W,H)
        avg_features = avg.max(dim=1)[0]  # (B,F)
        return avg_features, wxh_features


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


# cmn
class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=5):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk
        self.attention = MultiheadAttention(d_model, h)
        self.normalize = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, layer_past=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # # query1, att = self.attention(query, query)
        # # query2, att = self.attention(cross, query1)
        #
        # query1, att = self.attention(query, query)
        # cross1, att = self.attention(cross, cross)
        #
        # query2, att = self.attention(cross1, query1)
        # # cross2, att = self.attention(query1, cross1)
        #
        # query3 = query1 + query2
        #
        #
        # query, att = self.attention(query3, query)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


# --- Main Moduldes ---

class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                 fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()

        # For img & txt embedding and feature extraction
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None

        # For classification
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads)

        # Some constants
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)



    def forward(self, img=None, txt=None, lbl=None,retrie=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5,
                get_embed=False, get_txt_att=False):
        # --- Get img and txt features ---
        if img != None:  # (B,C,W,H) or ((B,V,C,W,H), (B,V))
            img_features, wxh_features = self.cnn(img)  # (B,F), (B,F,W,H)
            img_features = self.dropout(img_features)  # (B,F)

            # 将 retrie 转换为 PyTorch Tensor
            retrie_tensors = [torch.tensor(arr) for arr in retrie]

            # 使用 torch.stack 将列表中的 Tensor 堆叠起来，dim=0 表示在第一个维度上堆叠
            stacked_tensors = torch.stack(retrie_tensors, dim=0)

            stacked_tensors = stacked_tensors.float()

            # 使用 torch.mean 计算每个数组的平均值，dim=0 表示在第一个维度上计算平均值
            average_arrays = torch.mean(stacked_tensors, dim=0)


            retrieved_report = average_arrays
            retrieved_report = retrieved_report.int()


        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask)  # (B,L,E)

        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask)  # (B,L,E)

        # --- Fuse img and txt features ---
        if img != None and (txt != None or txt_embed != None):
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
            state_embed = self.state_embedding(state_index)  # (B,C,E)

            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics,
                                                                -1)  # (B,F) --> (B,T*E) --> (B,T,E)
            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask)  # (B,T,E), (B,T,L)
            final_embed = self.normalize(img_features + txt_features)  # (B,T,E)



        elif img != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
            state_embed = self.state_embedding(state_index)  # (B,C,E)

            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics,
                                                                -1)  # (B,F) --> (B,T*E) --> (B,T,E)
            final_embed = img_features  # (B,T,E)

        elif txt != None or txt_embed != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(txt_features.shape[0], 1).to(
                txt_features.device)  # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(txt_features.shape[0], 1).to(
                txt_features.device)  # (B,C)
            topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
            state_embed = self.state_embedding(state_index)  # (B,C,E)

            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask)  # (B,T,E), (B,T,L)
            final_embed = txt_features  # (B,T,E)

        else:
            raise ValueError('img and (txt or txt_embed) must not be all none')

        # Classifier output
        emb, att = self.attention(state_embed, final_embed)  # (B,T,E), (B,T,C)

        if lbl != None:  # Teacher forcing
            emb = self.state_embedding(lbl)  # (B,T,E)
        else:
            emb = self.state_embedding((att[:, :, 1] > threshold).long())  # (B,T,E)

        if get_embed:
            return att, final_embed + emb, retrieved_report  # (B,T,C), (B,T,E)
        elif get_txt_att and (txt != None or txt_embed != None):
            return att, txt_attention  # (B,T,C), (B,T,L)
        else:
            return att  # (B,T,C)


class Generator(nn.Module):
    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, num_tokens, num_posits, embed_dim=128, num_heads=1, fwd_dim=256, dropout=0.1, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.transform = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_tokens = num_tokens
        self.num_posits = num_posits

        self.d_model = 256
        self.num_heads = num_heads
        self.d_ff = 512
        self.dropout = dropout
        self.num_layers = num_layers

        self.cmn = MultiThreadMemory(num_heads, d_model=256, topk=5)
        self.model = self.make_model(num_tokens + 1, self.cmn)
        self.logit = nn.Linear(999, 116)
        self.linear = nn.Linear(999, 116)



    def forward(self, source_embed, token_index=None, source_pad_mask=None, target_pad_mask=None, max_len=300, top_k=1,
                bos_id=1, pad_id=3, mode='eye', memory_matrix=None, retrieved_index=None):
        if token_index != None:  # --- Training/Testing Phase ---#token_index=caption
            # Adding token embedding and posititional embedding.
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)  # (1,L) --> (B,L)
            posit_embed = self.posit_embedding(posit_index)  # (B,L,E)
            token_embed = self.token_embedding(token_index)  # (B,L,E)

            # # Memory querying and responding for textual features
            # final_memory_matrix = memory_matrix.unsqueeze(0).expand(token_embed.size(0),
            #                                                            memory_matrix.size(0),
            #                                                            memory_matrix.size(1))
            # responses = self.cmn(token_embed, final_memory_matrix, final_memory_matrix)
            # token_embed = token_embed + responses
            # # Memory querying and responding for textual features

            # retrieved_emb = self.token_embedding(retrieved_index)
            # token_embed, att = self.attention(retrieved_emb, token_embed)  # (B,T+L,E), (B,T+L,K)

            target_embed = token_embed + posit_embed  # (B,L,E)

            # Make embedding, attention mask, pad mask for Transformer Decoder
            final_embed = torch.cat([source_embed, target_embed], dim=1)  # (B,T+L,E)
            if source_pad_mask == None:
                source_pad_mask = torch.zeros((source_embed.shape[0], source_embed.shape[1]),
                                              device=source_embed.device).bool()  # (B,T)
            if target_pad_mask == None:
                target_pad_mask = torch.zeros((target_embed.shape[0], target_embed.shape[1]),
                                              device=target_embed.device).bool()  # (B,L)
            pad_mask = torch.cat([source_pad_mask, target_pad_mask], dim=1)  # (B,T+L)
            att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], target_embed.shape[1],
                                                                        mode).to(final_embed.device)  # (T+L,T+L)

            # Transformer Decoder

            # # Memory querying and responding for textual features
            # final_memory_matrix = self.memory_matrix.unsqueeze(0).expand(final_embed.size(0),
            #                                                            self.memory_matrix.size(0),
            #                                                            self.memory_matrix.size(1))
            # responses = self.cmn(final_embed, final_memory_matrix, final_memory_matrix)
            # final_embed = final_embed + responses
            # # Memory querying and responding for textual features
            att_masks = source_embed.new_ones(source_embed.shape[:2], dtype=torch.long)

            att_masks = att_masks.unsqueeze(-2)
            seq = token_index
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            # seq = self.linear(seq)

            out = self.model(source_embed, seq, att_masks, seq_mask, memory_matrix=memory_matrix)

            retrieved_emb = self.token_embedding(retrieved_index)
            # retrieved_emb, att = self.attention(retrieved_emb, retrieved_emb)  # (B,T+L,E), (B,T+L,K)
            # retrieved_emb, _ = self.attention(out, retrieved_emb)  # (B,T+L,E), (B,T+L,K)
            if out.shape[1] > retrieved_emb.shape[1]:
                padding = torch.zeros(
                    (retrieved_emb.shape[0], out.shape[1] - retrieved_emb.shape[1], retrieved_emb.shape[2]),
                    device=retrieved_emb.device)
                retrieved_emb = torch.cat([retrieved_emb, padding], dim=1)
            elif out.shape[1] < retrieved_emb.shape[1]:
                retrieved_emb = retrieved_emb[:, :out.shape[1], :]
            diff = out - retrieved_emb
            out = out + diff

            # out, _ = self.attention(retrieved_emb, out)  # (B,T+L,E), (B,T+L,K)

            # out=out*0.5+retrieved_emb*0.5

            final_embed = torch.cat([source_embed, out], dim=1)  # (B,T+L,E)

            # for i in range(len(self.transform)):
            #     final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]

            # Make prediction for next tokens
            token_index = torch.arange(self.num_tokens).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)  # (1,K) --> (B,K)
            token_embed = self.token_embedding(token_index)  # (B,K,E)
            emb, att = self.attention(token_embed, final_embed)  # (B,T+L,E), (B,T+L,K)

            # Truncate results from source_embed
            emb = emb[:, source_embed.shape[1]:, :]  # (B,L,E)
            att = att[:, source_embed.shape[1]:, :]  # (B,L,K)
            return att, emb
        else:  # --- Inference Phase ---
            return self.infer(source_embed, source_pad_mask, max_len, top_k, bos_id, pad_id, memory_matrix,retrieved_index)

    def infer(self, source_embed, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3, memory_matrix=None,
              retrieved_index=None):
        outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(
            source_embed.device) * bos_id  # (K,B,1) <s>
        scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device)  # (K,B)

        for _ in range(1, max_len):
            possible_outputs = []
            possible_scores = []

            for k in range(top_k):
                output = outputs[k]  # (B,L)
                score = scores[k]  # (B)

                att, emb = self.forward(source_embed, output, source_pad_mask=source_pad_mask,
                                        target_pad_mask=(output == pad_id), memory_matrix=memory_matrix,
                                        retrieved_index=retrieved_index)
                val, idx = torch.topk(att[:, -1, :], top_k)  # (B,K)
                log_val = -torch.log(val)  # (B,K)

                for i in range(top_k):
                    new_output = torch.cat([output, idx[:, i].view(-1, 1)], dim=-1)  # (B,L+1)
                    new_score = score + log_val[:, i].view(-1)  # (B)
                    possible_outputs.append(new_output.unsqueeze(0))  # (1,B,L+1)
                    possible_scores.append(new_score.unsqueeze(0))  # (1,B)

            possible_outputs = torch.cat(possible_outputs, dim=0)  # (K^2,B,L+1)
            possible_scores = torch.cat(possible_scores, dim=0)  # (K^2,B)

            # Pruning the solutions
            val, idx = torch.topk(possible_scores, top_k, dim=0)  # (K,B)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)  # (K,B)
            outputs = possible_outputs[idx, col_idx]  # (K,B,L+1)
            scores = possible_scores[idx, col_idx]  # (K,B)

        val, idx = torch.topk(scores, 1, dim=0)  # (1,B)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)  # (K,B)
        output = outputs[idx, col_idx]  # (1,B,L)
        score = scores[idx, col_idx]  # (1,B)
        return output.squeeze(0)  # (B,L)

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one':  # model can look at surrounding positions of the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye':  # model can only look at the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else:  # model can look at surrounding positions of the current index ith with some patterns
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz, sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# --- Full Models ---
class ClsGen(nn.Module):
    def __init__(self, classifier, generator, num_topics, embed_dim, num_heads):
        super().__init__()
        self.classifier = classifier
        self.generator = generator
        self.label_embedding = nn.Embedding(num_topics, embed_dim)
        self.cmn = MultiThreadMemory(num_heads, d_model=256, topk=5)
        self.memory_matrix = nn.Parameter(torch.FloatTensor(21, 256))
        nn.init.normal_(self.memory_matrix, 0, 1 / 256)
        self.normalize = nn.LayerNorm(256)
        self.attention = MultiheadAttention(embed_dim, num_heads)
        # self.token_embedding = nn.Embedding(num_tokens, embed_dim)




    def forward(self, image, history=None, caption=None, label=None, retrieval=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3,
                max_len=300, get_emb=False):
        label = label.long() if label != None else label
        img_mlc, img_emb, retri_txt = self.classifier(img=image, txt=history, lbl=label, retrie= retrieval, threshold=threshold, pad_id=pad_id,
                                           get_embed=True)  # (B,T,C), (B,T,E)
        lbl_idx = torch.arange(img_emb.shape[1]).unsqueeze(0).repeat(img_emb.shape[0], 1).to(img_emb.device)  # (B,T)
        lbl_emb = self.label_embedding(lbl_idx)  # (B,T,E)
        # Memory querying and responding for visual features
        memory_matrix = self.memory_matrix.unsqueeze(0).expand(img_emb.size(0), self.memory_matrix.size(0),
                                                               self.memory_matrix.size(1))
        # # not_predictions = 1 - img_mlc
        # #
        # # converted_predictions = torch.argmax(not_predictions, dim=-1)
        #
        # converted_predictions = img_mlc[:, :, 0]
        # # converted_predictions = label
        #
        # # memory_matrix = self.memory_matrix.unsqueeze(0).expand(img_emb.size(0), self.memory_matrix.size(0),
        # #                                                        self.memory_matrix.size(1))
        #
        # # expanded_tensor = torch.zeros(8, 21, 256)
        #
        # # 创建一个不需要梯度的副本
        # detached_label = converted_predictions.unsqueeze(2).detach()
        #
        # # 创建一个新的张量，并将 detached_label 的值复制给它
        # new_memory_matrix = memory_matrix.clone()
        # new_memory_matrix[:, :, :converted_predictions.size(1)] = detached_label
        #
        # # 使用新的张量替换原始的 memory_matrix
        # memory_matrix = new_memory_matrix
        # memory_matrix = self.normalize(memory_matrix)

        img_responses = self.cmn(img_emb, memory_matrix, memory_matrix)

        img_emb = img_emb + img_responses
        # Memory querying and responding for visual features

        # retrieved_emb = self.token_embedding(retri_txt)
        # out, att = self.attention(retrieved_emb, out)  # (B,T+L,E), (B,T+L,K)

        if caption != None:
            src_emb = img_emb + lbl_emb

            pad_mask = (caption == pad_id)
            cap_gen, cap_emb = self.generator(source_embed=src_emb, token_index=caption,
                                              target_pad_mask=pad_mask,
                                              memory_matrix=self.memory_matrix,
                                              retrieved_index=retri_txt)  # (B,L,S), (B,L,E)
            # cap_emb, att = self.attention(retrieved_emb, cap_emb)  # (B,T+L,E), (B,T+L,K)

            if get_emb:
                return cap_gen, img_mlc, cap_emb
            else:
                return cap_gen, img_mlc
        else:
            src_emb = img_emb + lbl_emb

            cap_gen = self.generator(source_embed=src_emb, token_index=caption, max_len=max_len, bos_id=bos_id,
                                     pad_id=pad_id, memory_matrix=self.memory_matrix, retrieved_index=retri_txt)  # (B,L,S)
            # cap_gen, att = self.attention(retrieved_emb, cap_gen)  # (B,T+L,E), (B,T+L,K)

            return cap_gen, img_mlc


class ClsGenInt(nn.Module):
    def __init__(self, clsgen, interpreter, freeze_evaluator=True):
        super().__init__()
        self.clsgen = clsgen
        self.interpreter = interpreter

        # Freeze evaluator's paramters
        if freeze_evaluator:
            for param in self.interpreter.parameters():
                param.requires_grad = False


    def forward(self, image, history=None, caption=None, label=None, retrieval=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3,
                max_len=300):
        if caption != None:
            pad_mask = (caption == pad_id)
            cap_gen, img_mlc, cap_emb = self.clsgen(image, history, caption, label, retrieval, threshold, bos_id, eos_id, pad_id,
                                                    max_len, True)
            cap_mlc = self.interpreter(txt_embed=cap_emb, pad_mask=pad_mask)
            return cap_gen, img_mlc, cap_mlc
        else:
            return self.clsgen(image, history, caption, label, retrieval, threshold, bos_id, eos_id, pad_id, max_len, False)
