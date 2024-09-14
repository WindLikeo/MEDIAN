import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
import os
from typing import Optional


from fusion.self_fusion import FourierSelfFusion

from GAT import GATLayer
import copy

def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class OPT:
    def __init__(self):
        self.hidden_dim = 512

class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2

class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, input_graph):
        hidden_states = input_graph
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states)
        return hidden_states  # B, seq_len, D

class MappingNet(nn.Module):
    def __init__(self, dim=512, select_channel=4):
        super().__init__()
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, select_channel)
        )
    def forward(self, x):
        new_x = x.transpose(2,1)
        new_x = self.weight(new_x)
        x = new_x.transpose(2,1)
        return x

class JointSemantic(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt.hidden_dim
        self.first_concat_gat = GAT(GATopt(self.hidden_dim, 1))

    def forward(self, raw_feature):

        first_graph_fea = self.first_concat_gat(raw_feature)
        vertex_fea = first_graph_fea

        emb_fea = l2norm(vertex_fea)
        return emb_fea

class Backbone(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, token_num=8, aggr_strategy=0, lambda_softmax=4):
        super().__init__()
        clip_path = ''
        self.clip, _, _ = open_clip.create_model_and_transforms('', pretrained=os.path.join(clip_path, ''))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.opt = OPT()
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(768, 512)
        self.text_fc = nn.Linear(512, 512)

        self.aggr_strategy = aggr_strategy

        self.image_local_map_net = MappingNet(dim=50, select_channel=token_num)
        self.text_local_map_net = MappingNet(dim=77, select_channel=token_num)

        self.image_fourier_aggr = FourierSelfFusion(using_ff_residual=False)
        self.text_fourier_aggr = FourierSelfFusion(using_ff_residual=False)

        self.getImageMediateFeature = JointSemantic(opt=self.opt)
        self.getTextMediateFeature = JointSemantic(opt=self.opt)

    def visual_out(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x)
        pooled, tokens = self.clip.visual._global_pool(x)
        # print(tokens.shape)
        pooled = pooled @ self.clip.visual.proj

        return pooled, x

    def text_out(self, text):
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        pooled, tokens = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                pooled = self.clip.text_projection(x)
            else:
                pooled = pooled @ self.clip.text_projection

        return pooled, x

    def extract_img_fea(self, x):
        img_global_fea, img_local_fea = self.visual_out(x)
        img_global_fea = img_global_fea.unsqueeze(1)

        img_local_fea = self.fc(img_local_fea.float())
        img_local_fea = self.image_local_map_net(img_local_fea)
        img_aggr_fea = self.image_fourier_aggr(img_local_fea.float())
        img_mediate_fea = torch.cat([img_aggr_fea, img_local_fea], dim=1)
        img_mediate_fea = self.getImageMediateFeature(img_mediate_fea)
        img_embs = img_local_fea

        final_img_feature = torch.cat([img_global_fea, img_mediate_fea, img_embs], dim=1)
        return final_img_feature, img_mediate_fea

    def extract_img_fea_patch_selection(self, img_x, txt):
        img_global_fea, img_local_fea = self.visual_out(img_x)
        img_global_fea = img_global_fea.unsqueeze(1)

        img_local_fea = self.fc(img_local_fea.float())
        img_local_fea = self.image_local_map_net(img_local_fea)

        img_aggr_fea = self.image_fourier_aggr(img_local_fea.float())

        txt_token = self.tokenizer(txt).cuda()
        text_global_fea, text_local_fea = self.text_out(txt_token)
        text_global_fea = text_global_fea.unsqueeze(1)

        text_local_fea = self.text_fc(text_local_fea.float())
        text_local_fea = self.text_local_map_net(text_local_fea)

        text_aggr_fea = self.text_fourier_aggr(text_local_fea.float())

        img_mediate_fea = torch.cat([img_aggr_fea, img_local_fea], dim=1)
        text_mediate_fea = torch.cat([text_aggr_fea, text_local_fea], dim=1)

        img_mediate_fea = self.getImageMediateFeature(img_mediate_fea)
        text_mediate_fea = self.getTextMediateFeature(text_mediate_fea)

        img_embs = img_local_fea
        final_img_feature = torch.cat([img_global_fea, img_mediate_fea, img_embs], dim=1)
        cap_embs = text_local_fea
        final_text_feature = torch.cat([text_global_fea, text_mediate_fea, cap_embs], dim=1)
        return final_img_feature, final_text_feature, img_mediate_fea, text_mediate_fea

    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).cuda()
        global_fea, x = self.text_out(txt)
        global_fea = global_fea.unsqueeze(1)
        x = self.text_fc(x.float())

        global_tokens = torch.matmul(self.chanel_score_text(global_fea.transpose(1, 2)),
                                     global_fea)  # self.global_attention(global_fea.unsqueeze(1))
        local_tokens = torch.matmul(self.chanel_score_text(x.transpose(1, 2)), x)
        return torch.cat([global_tokens, local_tokens], dim=1), (global_fea, x)


class FeatureWiseAffine(nn.Module):

    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()

        self.use_affine_level = use_affine_level

        self.MLP = nn.Sequential(
            nn.LayerNorm(in_channels * 2),
            nn.Linear(in_channels * 2, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed_ = torch.cat([x, text_embed], dim=-1)
        batch = x.shape[0]
        chanel = x.shape[1] * 2
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed_).reshape(batch, chanel, -1).chunk(2, dim=1)
            x = gamma * x + beta * text_embed
        return x


class MEDIAN(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, token_num=8, t=10, aggr_strategy=0, lambda_softmax=4):
        super().__init__()
        self.backbone = Backbone(hidden_dim, dropout, token_num, aggr_strategy=aggr_strategy, lambda_softmax=lambda_softmax)
        self.loss_T = nn.Parameter(torch.tensor([10.]))

        # self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num + 1)]))
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(token_num*3 + 1)]))

        self.t = t

        self.affine = FeatureWiseAffine(hidden_dim, hidden_dim, use_affine_level=True)

        self.aggr_strategy = aggr_strategy


    def target_fea(self, tag):
        tag_token, tag_mediate = self.backbone.extract_img_fea(tag)
        return tag_token, tag_mediate # , ref_mask

    def compose_feature(self, ref, mod):
        ref_token, mod_token, ref_mediate, mod_mediate = self.backbone.extract_img_fea_patch_selection(
            ref, mod)

        fuse_local = self.affine(ref_token, mod_token)

        return fuse_local, ref_token, mod_token, ref_mediate, mod_mediate

    def extract_retrieval_compose(self, ref, mod):

        fuse_local, _, _, _, _ = self.compose_feature(ref, mod)

        fuse_local = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)

        return fuse_local

    def extract_retrieval_target(self, tag):
        tag_local, _ = self.target_fea(tag)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local

    def compute_loss(self, ref, mod, tag):
        fuse_local, ref_token, mod_token, ref_mediate, mod_mediate = self.compose_feature(ref, mod)

        tag_local, tag_mediate = self.target_fea(tag)

        loss = {}

        retrieval_query = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)
        retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)

        ref_mediate = F.normalize(torch.mean(ref_mediate, dim=1), p=2, dim=-1)
        mod_mediate = F.normalize(torch.mean(mod_mediate, dim=1), p=2, dim=-1)
        tag_mediate = F.normalize(torch.mean(tag_mediate, dim=1), p=2, dim=-1)

        loss['stu_rank'] = self.info_nce(retrieval_query, retrieval_target)
        loss['kl'] = self.kl_div(ref_mediate, tag_mediate, mod_mediate, tag_mediate, self.t)
        return loss

    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl

