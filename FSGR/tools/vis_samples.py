#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, random, json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # project root

from data import ImageDetectionsField, TextField, RawField, COCO, DataLoader
from models.fsgr import TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from models.fsgr.transformer import Transformer


def _try_font(size=20):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _tensor_to_pil(img_tensor):
    """
    尝试把 dataloader 里的 [1,3,H,W] 或 [3,H,W] tensor 反归一化成可视图像。
    这里按 CLIP 默认 mean/std 做近似反归一化（仅用于可视化，不影响模型推理）。
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().float().cpu()

    # 先尝试 CLIP 反归一化（ViT-B/16 常用）
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    try:
        img = img * std + mean
    except Exception:
        pass

    img = img.clamp(0, 1)
    img = (img * 255.0).byte().permute(1,2,0).numpy()
    return Image.fromarray(img)


def _draw_caption_on_pil(pil_img, caption, pad_h=60):
    W, H = pil_img.size
    canvas = Image.new('RGB', (W, H + pad_h), (0, 0, 0))
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _try_font(20)
    draw.text((5, H + 5), caption, fill=(255, 255, 255), font=font)
    return canvas


def _flex_pick_images_and_id(img_pack):
    """
    从 image_dictionary 的第一个元素里尽可能提取出：
    - images: torch.Tensor [B,3,H,W]
    - img_ids: （可选）列表/张量，里面可能是字符串文件名或 int id

    兼容以下几种结构：
    1) img_pack = (images, _)
    2) img_pack = (images, ids) / (ids, images) / (images, scales, ids) 等
    3) img_pack = images（直接是 tensor）
    """
    images = None
    img_ids = None

    if torch.is_tensor(img_pack):
        images = img_pack
        return images, img_ids

    if isinstance(img_pack, (list, tuple)):
        # 在子元素里找 4D 图像张量与潜在的 ids
        for x in img_pack:
            if torch.is_tensor(x) and x.ndim == 4 and x.shape[1] in (1, 3):
                images = x
            elif isinstance(x, (list, tuple)):
                # 可能是 list 的文件名或 int id
                if len(x) > 0 and isinstance(x[0], (str, int)):
                    img_ids = x
            elif isinstance(x, torch.Tensor) and x.ndim == 1:
                # 可能是 1D 的 int id
                img_ids = x.tolist()

    return images, img_ids


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path',       type=str, default='./datasets/coco/images')
    parser.add_argument('--annotation_folder',   type=str, default='./m2_annotations')
    parser.add_argument('--pre_vs_path',         type=str, default='./.cache/clip/ViT-B-16.pt')
    parser.add_argument('--text_embed_path',     type=str, default='./text_embeddings/ram_ViT16_clip_text.pth')
    parser.add_argument('--ckpt',                type=str, default='./save_models/fsgr_fix_last.pth')
    parser.add_argument('--split',               type=str, default='val2014')   # val2014 / train2014
    parser.add_argument('--num',                 type=int, default=16)
    parser.add_argument('--out_dir',             type=str, default='./samples_val')
    parser.add_argument('--beam',                type=int, default=5)
    parser.add_argument('--max_len',             type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 构建 data pipeline（和训练一致）=====
    image_field = ImageDetectionsField(
        detections_path=args.features_path,
        labels_path=None,
        max_detections=49,
        load_in_tmp=False
    )
    text_field  = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True,
        tokenize='spacy', remove_punctuation=True, nopoints=False
    )

    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    _, val_dataset, _ = dataset.splits

    # 载入已有词表
    import pickle
    text_field.vocab = pickle.load(open('./vocab_language/vocab.pkl', 'rb'))

    # ===== 构建模型（与训练一致）=====
    encoder = TransformerEncoder(
        2, 0, text=False,
        attention_module=ScaledDotProductAttention,
        attention_module_kwargs={'m': 40}
    )
    decoder = TransformerDecoderLayer(
        len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>']
    )
    model = Transformer(
        text_field.vocab.stoi['<bos>'], encoder, decoder,
        [6, 11], pre_vs_path=args.pre_vs_path,
        text_emb_path=args.text_embed_path, pre_name="ViT-B/16",
        text=False, return_index=False
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    print("load pretrained weights!")
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # ===== 用 image_dictionary 保证预处理一致 =====
    dict_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dl = DataLoader(dict_val, batch_size=1, shuffle=True, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)

    # 采样 num 个 batch
    print(f"[viz] sampling {args.num} images from {args.split}")
    it = iter(dl)
    saved = 0
    pbar = tqdm(total=args.num, ncols=100)

    while saved < args.num:
        try:
            batch = next(it)  # 兼容 ((images, meta...), caps) 等
        except StopIteration:
            # 重新迭代
            it = iter(dl)
            continue

        # batch 结构容错解析
        if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
            # 不符合 image_dictionary 结构，直接跳过
            continue

        img_pack, _caps_gt = batch
        images, img_ids = _flex_pick_images_and_id(img_pack)

        if images is None or not torch.is_tensor(images):
            # 找不到图像张量，跳过
            continue

        images = images.to(device, non_blocking=True)

        # 生成
        try:
            out, _ = model.beam_search(
                images, args.max_len, text_field.vocab.stoi['<eos>'],
                args.beam, out_size=1
            )
        except Exception as e:
            # 个别 batch 异常，跳过
            print(f"[warn] beam_search failed on one sample: {e}")
            continue

        caps = text_field.decode(out, join_words=True)
        caption = caps[0] if isinstance(caps, list) and len(caps) > 0 else str(caps)

        # 尝试获得原图路径；如果没有 id，就用反归一化的 tensor 画图
        img_path = None
        if img_ids is not None:
            # id 可能是 ["COCO_val2014_000000xxxxxx.jpg"] 或 int
            img_id0 = img_ids[0]
            if isinstance(img_id0, str):
                img_path_try = os.path.join(args.features_path, args.split, img_id0)
                if os.path.isfile(img_path_try):
                    img_path = img_path_try
            elif isinstance(img_id0, int):
                # 如需从 COCO 字典映射，可在此处加入 json 反查；如果没有就跳过用 tensor 可视化
                pass

        if img_path and os.path.isfile(img_path):
            pil = Image.open(img_path).convert('RGB')
        else:
            # 用 dataloader 的图像 tensor 反归一化回显
            pil = _tensor_to_pil(images)

        canvas = _draw_caption_on_pil(pil, caption)
        out_name = f"sample_{saved:02d}.jpg"
        canvas.save(os.path.join(args.out_dir, out_name))
        saved += 1
        pbar.update(1)

    pbar.close()
    print(f"[viz] done. samples saved to: {args.out_dir}")
    print(f"示例保存在：{os.path.abspath(args.out_dir)}")
    

if __name__ == '__main__':
    main()
