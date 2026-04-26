import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from loralib import apply_lora, mark_only_lora_as_trainable, get_lora_parameters

from fs.utils.eval_utils import tokenize_texts, cls_acc
from fs.utils.model_utils import trainable_norm_params, trainable_bias_params, num_params


def prepare_for_first_stage(clip_model, args):
    vision_start = args.ln_vision_start
    text_start = args.ln_text_start
    if args.peft == "ln":
        trainable_params = trainable_norm_params(
            clip_model, 
            modality=args.ln_modality, 
            text_start=text_start, 
            vision_start=vision_start
        )
    elif args.peft == "lora":
        _ = apply_lora(args, clip_model, verbose=False)
        mark_only_lora_as_trainable(clip_model)
        trainable_params = get_lora_parameters(clip_model)
    
    elif args.peft == "bitfit":
        trainable_params = trainable_bias_params(
            clip_model, 
            modality=args.encoder, 
            text_start=text_start, 
            vision_start=vision_start
        )
    
    return clip_model, trainable_params



class SingleStreamClassifier(nn.Module):
    def __init__(self, clip_model, classnames, template="a photo of a {}."):
        super(SingleStreamClassifier, self).__init__()

        # let's keep a reference to the clip model (we will use it in the infer method)
        self.clip_model = clip_model
        
        # and also a reference to the visual backbone (handy for training code)
        self.backbone = clip_model.visual
        
        # linear classifier initialized with the text features of CLIP
        self._init_classifier(template, classnames)

        # also inherit the default temperature without changing it
        self.logit_scale = clip_model.logit_scale

        # create a map from category to embedding
        self.cat2id = {cat: i for i, cat in enumerate(classnames)}


    def _init_classifier(self, template, classnames):
        with torch.no_grad():
            texts = tokenize_texts(template=template, classnames=classnames)
            with torch.amp.autocast('cuda'):
                class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        self.classifier = nn.Parameter(class_embeddings.float(), requires_grad=True)


    def forward(self, x, no_grad_backbone=False):
        backbone_context_manager = torch.no_grad if no_grad_backbone else nullcontext
        with backbone_context_manager():
            x = self.backbone(x) 
        x = x / x.norm(dim=-1, keepdim=True)
        
        classifier = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * x @ classifier.t()
        return logits
    

    @torch.no_grad()
    def infer(self, x: torch.Tensor,
               categories: list[str], 
               template: str = "a photo of {}.", 
               compute_classifier_once: bool = True):
        
        if compute_classifier_once and hasattr(self, "inference_classifier"):
            classifier = self.inference_classifier
        else:
            # category-level inference; we only embed the categories that are not in the classifier already
            missing_classnames = [cat for cat in categories if cat not in self.cat2id]
            texts = tokenize_texts(template=template, classnames=missing_classnames)
            with torch.amp.autocast('cuda'):
                new_embeddings = self.clip_model.encode_text(texts)
            cat2new = {cat: new for cat, new in zip(missing_classnames, new_embeddings)}
            
            # while for categories in the classifier, we simply pre-pick them
            cat2known = {cat: self.classifier[self.cat2id[cat], :] for cat in categories if cat in self.cat2id}

            # now we stack everything into a single classifier
            # NOTE: for the simplified and controlled scenario of base2novel / all2all Few-Shot Learning, 
            # you don't need this implementation; however, this will also work 
            # in real-world scenarios where the distinction is only revealed on a per-sample basis
            classifier = [cat2new[cat] if cat in cat2new else cat2known[cat] for cat in categories]        
            classifier = torch.stack(classifier).to(x)
            classifier = classifier / classifier.norm(dim=-1, keepdim=True)
            self.inference_classifier = classifier

        # process the visual input
        x = self.backbone(x) 
        x = x / x.norm(dim=-1, keepdim=True)

        # then use the updated classifier to predict the current samples
        logits = self.logit_scale.exp() * x @ classifier.t()
        return logits


def save_twostage(args, clip_model, classifier_param):
    # we capture only the parameters that were actually trained
    stage1_weights = {}
    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            stage1_weights[name] = param.data.cpu() # save to cpu to be device-agnostic

    save_data = {
        'stage1_peft': args.peft,
        'stage1_weights': stage1_weights,
        'stage2_classifier': classifier_param.data.cpu(),
        'metadata': vars(args)
    }

    # directory management (reusing the logic from save_lora)
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = os.path.join(args.save_path, backbone, args.dataset, f"{args.shots}shots", f"seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{args.filename}.pt")
    torch.save(save_data, save_path)
    print(f'Two-Stage model weights saved to {save_path}')


@torch.no_grad()
def evaluate_selective_inference(
    model: SingleStreamClassifier, 
    loader: torch.utils.data.DataLoader, 
    template: str, 
    classnames: list[str]
):
    if hasattr(model, "inference_classifier"):
        delattr(model, "inference_classifier")
    
    model.eval()
    acc = 0.
    tot_samples = 0
    
    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model.infer(
                images, 
                categories=classnames, 
                template=template, 
            )
        acc += cls_acc(logits, target) * len(logits)
        tot_samples += len(logits)
    
    acc /= tot_samples
    return acc


def train_epoch_second_stage(model, optimizer, scheduler, scaler, train_loader, count_iters, total_iters):
    model.train()
    acc_train = 0
    tot_samples = 0
    loss_epoch = 0.

    for i, (images, target) in enumerate(train_loader):
        
        # move data to GPU
        images, target = images.cuda(), target.cuda()

        # get both text and image features
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(images, no_grad_backbone=True)
        
        # compute loss and backward with scaling
        loss = F.cross_entropy(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
            count_iters += 1
        optimizer.zero_grad()
        
        # compute accuracy and loss
        acc_train += cls_acc(logits, target) * target.shape[0]
        loss_epoch += loss.item() * target.shape[0]
        tot_samples += target.shape[0]
        
        # check if we reached the total number of iterations
        if count_iters == total_iters:
            break
        
    # print after each epoch
    if count_iters <= total_iters:
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))

    return model, count_iters



def train_epoch(clip_model, optimizer, scheduler, scaler, train_loader, tokenized_texts, count_iters, total_iters, args):
    clip_model.train()
    acc_train = 0
    tot_samples = 0
    loss_epoch = 0.

    text_context_manager = torch.no_grad if args.ln_modality == 'vision' else nullcontext
    vision_context_manager = torch.no_grad if args.ln_modality == 'text' else nullcontext

    for i, (images, target) in enumerate(train_loader):
        
        # move data to GPU
        images, target = images.cuda(), target.cuda()

        # get both text and image features
        # wrapping the forward pass in autocast
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with text_context_manager():
                text_features = clip_model.encode_text(tokenized_texts)
            with vision_context_manager():
                image_features = clip_model.encode_image(images)
        
        # well, you know that clip normalizes the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # compute loss and backward with scaling
        cosine_similarity = clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(cosine_similarity, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
            count_iters += 1
        optimizer.zero_grad()
        
        # compute accuracy and loss
        acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
        loss_epoch += loss.item() * target.shape[0]
        tot_samples += target.shape[0]
        
        # check if we reached the total number of iterations
        if count_iters == total_iters:
            break
        
    # print after each epoch
    if count_iters <= total_iters:
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))

    return clip_model, count_iters



def run_twostage(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    # train only layer-norm instances
    clip_model, trainable_params = prepare_for_first_stage(clip_model, args)
    clip_model = clip_model.float().cuda() 
    print(f"Trainable parameters: {num_params(clip_model, trainable=True):,}")    
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    print(f"Using AdamW with lr={args.lr}, wd={args.wd}, betas=(0.9, 0.999).")
    
    total_iters = args.n_iters * args.shots
    cosine_iters = int(total_iters*args.n_iters_frac)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_iters, eta_min=1e-6)
    
    # training 
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0

    # we only need to tokenize once
    tokenized_texts = tokenize_texts(template=dataset.template[0], classnames=dataset.classnames)

    # start training for a fixed number of gradient steps (total_iters)  
    while count_iters < cosine_iters:
        clip_model, count_iters = train_epoch(
            clip_model, 
            optimizer, 
            scheduler, 
            scaler, 
            train_loader, 
            tokenized_texts, 
            count_iters, 
            cosine_iters,
            args
        )

        if args.debug: break
    
    # once the first stage is done, we freeze everything and exploit the learned layer-norms
    model = SingleStreamClassifier(clip_model, dataset.classnames, template=dataset.template[0])
    print("Initialized single stream classifier")
    optimizer = torch.optim.AdamW([model.classifier], lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    print(f"[Second Stage] Using AdamW with lr={args.lr}, wd={args.wd}, betas=(0.9, 0.999).")
    
    # training 
    scaler = torch.amp.GradScaler('cuda')
    second_stage_iters = total_iters - count_iters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, second_stage_iters, eta_min=1e-6)

    # training the classifier
    while count_iters < total_iters:
        model, count_iters = train_epoch_second_stage(
            model,
            optimizer,
            scheduler,
            scaler,
            train_loader,
            count_iters,
            total_iters
        )
        if args.debug: break


    # save the model if save_path is provided
    if args.save_path is not None:
        save_twostage(args, clip_model, model.classifier)


    # test after training
    if args.setting == "base2new":
        # we have packed both base/new dataloaders into a single var for convenience
        test_base_loader, test_new_loader = test_loader
        
        # evaluation on base classes
        acc_test_base = evaluate_selective_inference(
            model, test_base_loader, template=dataset.template[0], classnames=dataset.test_classnames
        )
        print("\n**** Final Test-Base accuracy: {:.2f}. ****\n".format(acc_test_base))
        
        # evaluation on the novel classes
        acc_test_new = evaluate_selective_inference(
            model, test_new_loader, template=dataset.template[0], classnames=dataset.test_new_classnames
        )
        print("**** Final Test-New accuracy: {:.2f}. ****\n".format(acc_test_new))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_new}
    else:
        # in this case, we are in a "all2all" setting, so we use the output of the 2nd stage
        acc_test = evaluate_selective_inference(
            model, test_loader, template=dataset.template[0], classnames=dataset.classnames
        )
        print("\n**** Final Test Accuracy (all categories): {:.2f}. ****\n".format(acc_test))
        result = {"acc_test": acc_test}
    
    # always return a dictionary with metric_name: float mappings so we can dump easily later on
    return result