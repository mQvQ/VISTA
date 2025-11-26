import torch
import torch.nn as nn
from functools import partial
import math
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
import open_clip

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from ldm.modules.diffusionmodules.PixArt_blocks import MultiHeadCrossAttention
from einops import rearrange
# from ldm.modules.x_transformer import (
#     Encoder,
#     TransformerWrapper,
# )  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device="cuda",
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


def clip_transformer_forward(model, input_ids_list, attention_mask, class_embed=None):
    # this is a hack to get the CLIP transformer to work with long captions
    # class_embed is concatenated to the input embeddings

    output_attentions = model.config.output_attentions
    output_hidden_states = model.config.output_hidden_states
    return_dict = model.config.use_return_dict

    sz = input_ids_list[0].size()
    input_shape = (sz[0], sz[1] * len(input_ids_list))

    hidden_states_list = []

    for input_ids in input_ids_list:
        hidden_states = model.embeddings(input_ids)
        hidden_states_list.append(hidden_states)

    hidden_states = torch.cat(hidden_states_list, dim=1)

    if class_embed is not None:
        input_shape = (input_shape[0], 1 + input_shape[1])
        class_embed = class_embed.unsqueeze(1)
        hidden_states = torch.cat([class_embed, hidden_states], dim=1)

    # causal mask is applied over the whole sequence (154 tokens)
    causal_attention_mask = _make_causal_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )

    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = model.final_layer_norm(last_hidden_state)

    return last_hidden_state


class FrozenCLIPEmbedder(nn.Module):
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.clip_max_length = self.tokenizer.model_max_length
        except:
            # when using plip model
            self.tokenizer = AutoTokenizer.from_pretrained(version)
            self.clip_max_length = max_length

        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = self.clip_max_length * math.ceil(
            max_length / self.clip_max_length
        )
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = batch_encoding["input_ids"].to(self.device)
        attention_mask = batch_encoding["attention_mask"].to(self.device)

        if input_ids.shape[1] != self.clip_max_length:
            input_ids_list = input_ids.split(self.clip_max_length, dim=-1)
        else:
            input_ids_list = [input_ids]

        z = clip_transformer_forward(self.transformer.text_model, input_ids_list, attention_mask)
        return z

    @torch.no_grad()
    def encode(self, text):
        return self(text)


class FrozenCONCHTextEmbedder(nn.Module):
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""
    # 固定
    def __init__(
            self,
            checkpoint_path='/home/tma/VISTA/CONCH/checkpoints/conch/pytorch_model.bin',
            version="conch_ViT-B-16",
            device="cuda",
    ):
        super().__init__()

        self.device = device
        self.tokenizer = get_tokenizer()

        self.model, preprocess = create_model_from_pretrained(version, checkpoint_path, device=device)
        _ = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # self.proj = nn.Linear(768, 512).to(device)


    def forward(self, text):
        tokenize_prompts = tokenize(texts=text, tokenizer=self.tokenizer).to(self.device)
        _, text_emb = self.model._encode_text(tokenize_prompts)
        # text_emb = self.proj(text_emb)

        return text_emb


class FrozenCONCHImageEmbedder(nn.Module):
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""
    # 固定
    def __init__(
            self,
            checkpoint_path='/home/tma/ST-Diffusion/CONCH/checkpoints/conch/pytorch_model.bin',
            version="conch_ViT-B-16",
            device="cuda",
    ):
        super().__init__()

        self.device = device

        self.model, self.preprocess = create_model_from_pretrained(version, checkpoint_path, device=device)
        _ = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, image):
        # preprocess 写在dataset
        image = image.to(self.device)
        _, image_emb = self.model._encode_image(image)

        return image_emb


class FrozenImageCLIP(nn.Module):
    '''encode image'''
    """Uses the openai CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
    ):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(version)  # for image
        self.transformer = CLIPVisionModel.from_pretrained(version).to(device)
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        for k, v in inputs.items(): inputs[k] = v.to(self.device)
        outputs = self.transformer(**inputs)
        # z = outputs.last_hidden_state bsz, 50, 768
        pooled_output = outputs.pooler_output # bsz, 768
        # print('clip encoder embed: z, pooled_output', z.shape, pooled_output.shape)
        return pooled_output

    def forward(self, images):
        return self.encode(images)



class BioMedCLIPEmbedder(nn.Module):
    """Uses microsoft Biomed CLIP transformer (from hf, based on openclip)
    has a max context length of 256
    """

    def __init__(self, device="cuda", max_length=77):
        super().__init__()
        version = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.clip, _, _ = open_clip.create_model_and_transforms(
            f"hf-hub:{version}"
        )
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{version}")

        self.max_length = max_length
        self.device = device
        self.freeze()

    def freeze(self):
        self.clip = self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, text):
        tokens = self.tokenizer(text, context_length=self.max_length).to(self.device)

        z = self.clip.text.transformer(tokens)[0]
        return z

    def forward(self, text):
        return self.encode(text)

class RNAEncoder(nn.Module):
    def __init__(self,
                 in_channels, hidden_dims, device='cuda', ckpt=None):
        super(RNAEncoder, self).__init__()

        self.in_channels = in_channels
        self.device = device
        modules = [
        nn.Sequential(nn.Dropout())]
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules).to(self.device)
        if ckpt:
            weights = refine_rna_encoder_weights(ckpt)
            incompatible_keys = self.encoder.load_state_dict(weights, strict=False)
            print("RNA encoder Missing keys:", incompatible_keys.missing_keys)
            print("RNA encoder Unexpected keys:", incompatible_keys.unexpected_keys)

        self.freeze()

    def freeze(self):
        self.encoder = self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        return self.encoder(x)


class SingleRNAEncoder(nn.Module):
    def __init__(self,
                 in_channels, hidden_dims, device='cuda', ckpt=None):
        super(SingleRNAEncoder, self).__init__()

        self.in_channels = in_channels
        self.device = device
        modules = [
        nn.Sequential(nn.Dropout())]
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules).to(self.device)
        if ckpt:
            weights = refine_rna_encoder_weights(ckpt)
            incompatible_keys = self.encoder.load_state_dict(weights, strict=False)
            print("RNA encoder Missing keys:", incompatible_keys.missing_keys)
            print("RNA encoder Unexpected keys:", incompatible_keys.unexpected_keys)

        self.freeze()
        # 768维映射到512维
        self.linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        ).to(self.device)


    def freeze(self):
        self.encoder = self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.linear(x)
        x = x.unsqueeze(1)
        return x


def refine_rna_encoder_weights(ckpt):
    loaded_weights = torch.load(ckpt)
    new_weights = {}
    for key, value in loaded_weights.items():
        # 修改键
        new_key = key.replace('encoder.encoder.', '')
        new_weights[new_key] = value
    return new_weights


class JointConditionEncoder(nn.Module):
    '''
    joint condition: text-rna-img
    '''
    def __init__(self, version, max_length, fusion_way='cat', device='cuda', rna_ckpt=None): #
        super().__init__()
        print('prepare encode')
        self.text_encoder = FrozenCLIPEmbedder(version=version, max_length=max_length, device=device)
        print('text-encode', self.text_encoder)
        self.image_encoder = FrozenImageCLIP(version=version, device=device)
        print('img-encode', self.image_encoder)
        # self.rna_encoder = RNAEncoder(18428, [4096, 2048], 768)
        self.rna_encoder = RNAEncoder(20531, [6000, 4000, 768], device=device, ckpt=rna_ckpt)
        if fusion_way == 'cat': # bsz, 154, 512+768+768
           self.fusion_module = nn.Sequential(
               nn.Linear(768*2+512, 1024),
               nn.ReLU(),
               nn.Linear(1024, 512)
           ).to(device)
        self.device = device
        self.text_encoder.freeze()
        self.image_encoder.freeze()
        self.freeze()

    def freeze(self):
        self.rna_encoder = self.rna_encoder.eval()
        for param in self.rna_encoder.parameters():
            param.requires_grad = False

    def encode(self, conds):
        x1 = self.text_encoder.encode(conds['caption']) # [bsz, max_length, 512]
        x2 = self.image_encoder.encode(conds['image']) # [bsz, 768]
        x3 = self.rna_encoder(conds['rna']) # [bsz, 768]
        # print('text-image-rna conds shape', x1.shape, x2.shape, x3.shape)
        bsz = x2.shape[0]
        token_num = x1.shape[1]
        x2 = x2.unsqueeze(1).repeat(1, token_num, 1)
        x3 = x3.unsqueeze(1).repeat(1, token_num, 1)
        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.fusion_module(x)
        return x

    def forward(self, conds):
        return self.encode(conds)

class TwoStageEncoder(nn.Module):
    '''
    context encoder:[bsz, text_token+image_token, token_dim]
    '''
    def __init__(self, text_cond_config, image_cond_config, only_img=False):
        super().__init__()
        self.text_embeder = FrozenCONCHTextEmbedder(**text_cond_config)
        self.image_embeder = FrozenCONCHImageEmbedder(**image_cond_config)
        self.only_img = only_img

    def forward(self, conds):
        image = conds['image']
        image_tokens = self.image_embeder(image)
        if self.only_img: return image_tokens

        text = conds['caption']
        text_tokens = self.text_embeder(text)
        x = torch.cat((text_tokens, image_tokens), dim=1)

        return x


class ImputationConditionEncoder(nn.Module):
    '''
    text+img+mask rna
    '''
    def __init__(self, text_cond_config, image_cond_config):
        super().__init__()
        self.text_embeder = FrozenCONCHTextEmbedder(**text_cond_config)
        self.image_embeder = FrozenCONCHImageEmbedder(**image_cond_config)
        self.rna_encoder = RNAEncoder(20531, [6000, 4000, 768], device=device, ckpt=rna_ckpt)


    def forward(self, conds):
        text = conds['caption']
        image = conds['image']
        rna = conds['rna']
        text_tokens = self.text_embeder(text)
        image_tokens = self.image_embeder(image)
        x = torch.cat((text_tokens, image_tokens), dim=1)

        return x


class CellTypeEncoder(nn.Module):
    # generate celltype prototypes and compose final celltypes tokens
    def __init__(self, num_celltypes, prototype_dim, device='cuda'):
        super(CellTypeEncoder, self).__init__()
        self.device = device
        self.celltype_emb_pos = nn.Embedding(num_celltypes, prototype_dim).to(self.device)
        self.celltype_emb_neg = nn.Embedding(num_celltypes, prototype_dim).to(self.device)
        self.num_celltypes = num_celltypes
        self.prototype_dim = prototype_dim

    def forward(self, conds):
        cell_ratios = conds.to(self.device) # B, num_celltypes=73
        # 获取每个celltype的正负向嵌入向量
        celltype_emb_pos = self.celltype_emb_pos.weight  # (num_celltypes, prototype_dim)
        celltype_emb_neg = self.celltype_emb_neg.weight  # (num_celltypes, prototype_dim)
        # 扩展cell_ratios以匹配嵌入向量的维度
        cell_ratios = cell_ratios.unsqueeze(-1)  # (bsz, num_celltypes, 1)
        # 计算每个celltype的嵌入向量
        celltype_emb = cell_ratios * celltype_emb_pos + (1 - cell_ratios) * celltype_emb_neg

        # 将结果展平为 (bsz, num_celltypes, prototype_dim)
        celltype_emb = celltype_emb.reshape(-1, self.num_celltypes, self.prototype_dim)

        return celltype_emb.to(torch.float32)



class MultiControlEncoderv3(nn.Module):
    '''
    image_tokens
    cell-type prototype tokens
    organ & cancer type: label_embeddings
    '''
    def __init__(self, image_cond_config, celltype_cond_config, num_classes, device='cuda', only_img=False):
        super().__init__()
        self.image_embeder = FrozenCONCHImageEmbedder(**image_cond_config)
        self.celltype_embeder = CellTypeEncoder(**celltype_cond_config)
        self.only_img = only_img
        self.device = device
        self.image_filter = MultiHeadCrossAttention(d_model=768, num_heads=8).to(self.device)
        self.norm1 = nn.LayerNorm(768, elementwise_affine=False, eps=1e-6).to(self.device)
        self.norm2 = nn.LayerNorm(768, elementwise_affine=False, eps=1e-6).to(self.device)
        self.ff = nn.Sequential(
            nn.Linear(768, 768 * 4),
            nn.ReLU(),
            nn.Linear(768*4, 768)
        )
        self.label_emb = nn.Embedding(num_classes+1, 192 * 4).to(self.device)
        self.num_classes = num_classes

    def forward(self, conds):
        image = conds['image']
        if 'image_token_mask' in conds:
            mask = conds['image_token_mask']
        else: mask=None
        image_tokens = self.image_embeder(image)
        if self.only_img: return image_tokens
        celltype = conds['celltype']
        celltype_tokens = self.celltype_embeder(celltype).to(torch.float32)
        image_tokens = self.norm1(image_tokens + self.image_filter(image_tokens, celltype_tokens, mask))
        x = torch.cat((celltype_tokens, image_tokens), dim=1)
        # add class_labels condition
        x = self.norm2(x + self.ff(x))
        y = conds['label'].to(self.device)
        y = torch.where(y == -1, torch.full_like(y, self.num_classes), y) # uncondition 的情况

        y = self.label_emb(y) #  # b, 6D
        cc = torch.concat([y, x], dim=1)
        return cc

class MultiControlEncoderv6(nn.Module):
    def __init__(self, image_cond_config, celltype_cond_config, num_classes, device='cuda', only_img=False):
        super().__init__()
        self.image_embeder = FrozenCONCHImageEmbedder(**image_cond_config)
        self.celltype_embeder = CellTypeEncoder(**celltype_cond_config)
        self.only_img = only_img
        self.device = device

        self.image_filter = MultiHeadCrossAttention(d_model=768, num_heads=8).to(self.device)

        # 分支各自的 Pre-Norm
        self.norm_img = nn.LayerNorm(768, eps=1e-6).to(self.device)
        self.norm_ct  = nn.LayerNorm(768, eps=1e-6).to(self.device)
        self.norm_lbl = nn.LayerNorm(192*4, eps=1e-6).to(self.device)

        # FF 走 Pre-Norm
        self.norm_ff  = nn.LayerNorm(768, eps=1e-6).to(self.device)
        self.ff = nn.Sequential(nn.Linear(768, 768*4), nn.ReLU(), nn.Linear(768*4, 768))

        # label embedding
        self.label_emb = nn.Embedding(num_classes+1, 192*4).to(self.device)
        self.num_classes = num_classes

        # 可学习门控，初值小，避免一上来扰动大
        self.g_cell  = nn.Parameter(torch.tensor(0.1))
        self.g_label = nn.Parameter(torch.tensor(0.1))

    def forward(self, conds):
        image = conds['image']
        mask = conds.get('image_token_mask', None)

        image_tokens = self.image_embeder(image)                 # [B, Ti, 768]
        if self.only_img: return image_tokens

        # celltype branch
        celltype = conds['celltype']                             # [B, C]
        celltype_tokens = self.celltype_embeder(celltype).to(torch.float32)  # [B, C, 768]

        # cross-attn: Pre-Norm
        attn_out = self.image_filter(self.norm_img(image_tokens), self.norm_ct(celltype_tokens), mask)
        image_tokens = image_tokens + attn_out                   # residual add

        # concat 前分别归一 + 缩放门
        y_idx = conds['label'].to(self.device)
        y_idx = torch.where(y_idx == -1, torch.full_like(y_idx, self.num_classes), y_idx)
        y = self.label_emb(y_idx)                                # [B, 192*4]
        y = self.g_label * self.norm_lbl(y)                      # [B, 192*4]

        celltype_tokens = self.g_cell * self.norm_ct(celltype_tokens)

        # 融合
        x = torch.cat((celltype_tokens, image_tokens), dim=1)    # [B, C+Ti, 768]
        x = x + self.ff(self.norm_ff(x))                         # Pre-Norm FFN

        cc = torch.cat([y, x], dim=1)                            # [B, 1+C+Ti, 768]
        return cc


class MultiControlEncoderv6_light(nn.Module):
    """
    Light-weight multi-conditional encoder:
    - Morphology tokens from frozen image encoder
    - Low-dimensional cell-type tokens (from CellTypeEncoder)
    - Low-dimensional cancer-type embedding
    """
    def __init__(
        self,
        image_cond_config,
        celltype_cond_config,
        num_classes,
        device="cuda",
        only_img=False,
        celltype_dim=64,     # reduced dim per cell type
        label_dim=64,       # reduced dim per label
        fuse_dim=256         # unified condition dim
    ):
        super().__init__()
        self.device = device
        self.only_img = only_img

        # frozen image encoder
        self.image_embeder = FrozenCONCHImageEmbedder(**image_cond_config)
        # cell-type encoder (73 × prototype_dim)
        self.celltype_embeder = CellTypeEncoder(**celltype_cond_config)

        # project high-dim celltype embeddings to low-dim
        self.cell_proj = nn.Linear(celltype_cond_config["prototype_dim"], celltype_dim).to(device)

        # label embedding (num_classes+1 to handle uncondition)
        self.label_emb = nn.Embedding(num_classes + 1, label_dim).to(device)
        self.num_classes = num_classes

        # unify all conditional tokens into a shared space (for diffusion backbone)
        self.fuse_proj = nn.Linear(celltype_dim, fuse_dim).to(device)
        self.label_proj = nn.Linear(label_dim, fuse_dim).to(device)
        self.image_proj = nn.Linear(768, fuse_dim).to(device)

        # optional gating and normalization
        self.norm_ct = nn.LayerNorm(celltype_dim)
        self.norm_lbl = nn.LayerNorm(label_dim)
        self.norm_img = nn.LayerNorm(768)

        self.g_cell = nn.Parameter(torch.tensor(0.5))
        self.g_label = nn.Parameter(torch.tensor(0.5))

        # lightweight cross-attention between image and cell-type
        self.image_filter = MultiHeadCrossAttention(d_model=fuse_dim, num_heads=4).to(device)
        self.ff = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, fuse_dim * 4),
            nn.ReLU(),
            nn.Linear(fuse_dim * 4, fuse_dim)
        ).to(device)

    def forward(self, conds):
        image = conds["image"]
        mask = conds.get("image_token_mask", None)
        image_tokens = self.image_embeder(image)
        if self.only_img: return image_tokens

        # ---- celltype branch ----
        celltype_ratios = conds["celltype"]  # (B, num_celltypes)
        celltype_tokens = self.celltype_embeder(celltype_ratios)  # (B, num_celltypes, prototype_dim)
        celltype_tokens = self.cell_proj(celltype_tokens)         # (B, num_celltypes, celltype_dim)
        celltype_tokens = self.g_cell * self.norm_ct(celltype_tokens)
        celltype_tokens = self.fuse_proj(celltype_tokens)

        # ---- label branch ----
        y_idx = conds["label"].to(self.device)
        y_idx = torch.where(y_idx == -1, torch.full_like(y_idx, self.num_classes), y_idx)
        y = self.label_emb(y_idx)                                 # (B, label_dim)
        y = self.g_label * self.norm_lbl(y)
        y = self.label_proj(y)                                    # (B, 1, fuse_dim)

        # ---- image branch ----
        image_tokens = self.norm_img(image_tokens)
        image_tokens = self.image_proj(image_tokens)              # (B, Ti, fuse_dim)

        # ---- fuse via cross-attn ----
        attn_out = self.image_filter(image_tokens, celltype_tokens, mask)
        fused_img = image_tokens + attn_out
        fused_img = fused_img + self.ff(fused_img)

        # ---- concatenate all conditions ----
        cc = torch.cat([y, celltype_tokens, fused_img], dim=1)    # (B, 1+C+Ti, fuse_dim)
        return cc


if __name__ == '__main__':
    import numpy as np
    device='cuda:0'
    # img = torch.zeros((256, 256, 3))
    # rna = torch.zeros(1, 20531).to(device)
    # ckpt = '/home/tma/PathLDM/models/betavae-tcga-log-norm/model_dict_best.pt'
    # conds = {'caption': 'test model', 'image': img, 'rna': rna}
    # # encoder = JointConditionEncoder(version="vinid/plip", max_length=154, device=device, rna_ckpt=ckpt)
    # # encoder.forward(conds)
    # ## test RNAEncoder
    # img = torch.zeros((256, 256, 3))
    # rna = torch.zeros(4, 20531).to(device)
    # conds = {'rna': rna}
    # encoder = SingleRNAEncoder(20531, [6000, 4000, 768], device=device, ckpt='/home/tma/PathLDM/models/betavae-tcga-log-norm/model_dict_best.pt')
    # encoder.forward(conds['rna'])
    texts = ['test conch', 'test2']
    # encoder = FrozenCONCHTextEmbedder(device='cuda:0')
    # cond_emb = encoder.forward(texts)
    # print(cond_emb.shape)
    # encoder = FrozenCLIPEmbedder(version="vinid/plip")
    # cond_emb = encoder.forward(texts)
    # print(cond_emb.shape)
    # image = torch.randn((2, 3, 256, 256))# after preprocess
    # conds = {'caption':texts, 'image': image}
    # encoder = TwoStageEncoder(text_cond_config={'device': 'cuda:0'}, image_cond_config={'device': 'cuda:0'})
    # emb = encoder.forward(conds)
    # print(emb.shape)
    image = torch.randn((2, 3, 256, 256)).cuda()
    celltype = torch.randn((2, 10)).cuda()
    label = torch.tensor([1, -1]).cuda()
    conds = {'conds': {'celltype': celltype, 'image': image, 'label': label}}
    encoder = MultiControlEncoder(image_cond_config={'device': 'cuda:0'},
                                    celltype_cond_config={'num_celltypes': 10, 'prototype_dim': 768, 'device': 'cuda:0'},
                                  num_classes=5).cuda()
    emb = encoder.forward(conds['conds'])
    print(emb[0].shape, emb[1].shape)