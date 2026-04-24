import os
import clip
import torch
import os.path as osp


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.to('cpu', non_blocking=True))
            labels.append(target.to('cpu', non_blocking=True))
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels


@torch.no_grad()
def zero_shot_eval(clip_model, dataset, loader, split="test"):
    assert split in ("train", "val", "test")
    # Textual features
    classnames = getattr(dataset, f"{split}_classnames") if split != "train" else dataset.classnames
    print("About to run clip_classifier", clip_model.visual.proj.device)
    textual_features = clip_classifier(classnames, dataset.template, clip_model)

    # Pre-load test features
    print("About to run pre_load_features", clip_model.visual.proj.device)
    test_features, test_labels = pre_load_features(clip_model, loader)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = clip_model.logit_scale.exp() * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's {} accuracy: {:.2f}. ****\n".format(split, zs_acc))

    # free-up memory
    del test_features, test_labels, textual_features
    torch.cuda.empty_cache()
    return zs_acc



@torch.no_grad()
def evaluate(clip_model, loader, template, classnames):
    clip_model.eval()
    texts = tokenize_texts(template, classnames)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(texts)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()
        acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
        tot_samples += len(cosine_similarity)
    
    acc /= tot_samples
    return acc


def tokenize_texts(template, classnames, device='cuda'):
    texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
    tokenized_texts = clip.tokenize(texts).to(device)
    return tokenized_texts 


def dump(result: dict, args: dict, decimals: int = 4):
    import pandas as pd
    from typing import Iterable

    args["backbone"] = args["backbone"].replace("/", "-")
    
    outpath = osp.join(args["results_dir"], args["exp_name"])
    if not outpath.endswith(".csv"): outpath += ".csv"
    os.makedirs(osp.dirname(outpath), exist_ok=True)

    result.update(args)
    result = {k: [v] for k, v in result.items()}
    df = pd.DataFrame.from_dict(result)

    for col in df.columns:
        if "acc" in str(col):
            df[col] = df[col].round(decimals)
    
    df.to_csv(outpath, index=False)
    print(f"Saved result at {outpath} =)")