import torch
from attack_implementations import Attack


def normalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _extract_model_outputs(model, images, out_idx=0):
    outputs = model(normalize_image(images))

    if isinstance(outputs, tuple):
        logits = outputs[out_idx]
    elif isinstance(outputs, dict):
        if 'out' in outputs:
            logits = outputs['out']
        else:
            logits = list(outputs.values())[0]
    else:
        logits = outputs

    return logits


def _build_margin_weight(outputs, labels, beta=1.0, ignore_index=255):
    probs = torch.softmax(outputs, dim=1)
    valid_mask = (labels != ignore_index)

    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0

    p_true = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)

    tmp_probs = probs.clone()
    tmp_probs.scatter_(1, safe_labels.unsqueeze(1), -1e9)
    p_other_max = tmp_probs.max(dim=1)[0]

    margin = p_true - p_other_max
    margin_pos = torch.clamp(margin, min=0.0)
    margin_weight = torch.exp(1.5 * margin_pos)
    margin_weight = margin_weight * valid_mask.float()

    return margin_weight, margin


def _build_phase_aware_margin_weight(
    outputs,
    labels,
    raw_margin_weight,
    iteration,
    iterations,
    gamma_min=0.2,
    ignore_index=255
):
    pred = outputs.argmax(dim=1)
    valid_mask = (labels != ignore_index)
    state_mask = ((pred == labels) & valid_mask).float()

    progress = float(iteration + 1) / float(iterations)
    gamma_t = max(gamma_min, 1.0 - progress)

    gated_margin_weight = (
        1.0
        + state_mask * (raw_margin_weight - 1.0)
        + (1.0 - state_mask) * gamma_t * (raw_margin_weight - 1.0)
    )
    gated_margin_weight = gated_margin_weight * valid_mask.float()

    return gated_margin_weight, state_mask, gamma_t


def _build_high_margin_mask(margin, labels, pred, top_ratio=0.2, ignore_index=255):
    valid_mask = (labels != ignore_index)
    correct_mask = (pred == labels) & valid_mask

    margin_for_select = margin.clone()
    margin_for_select[~correct_mask] = -1e9

    high_margin_mask = torch.zeros_like(margin, dtype=torch.float)

    for b in range(margin.shape[0]):
        correct_margin = margin_for_select[b][correct_mask[b]]
        if correct_margin.numel() == 0:
            continue

        k = max(1, int(top_ratio * correct_margin.numel()))
        threshold = torch.topk(correct_margin, k=k).values.min()
        high_margin_mask[b] = ((margin_for_select[b] >= threshold) & correct_mask[b]).float()

    return high_margin_mask, correct_mask.float()


def _build_uncertainty_weight(
    outputs,
    labels,
    gamma=0.5,
    max_weight=1.5,
    ignore_index=255
):
    probs = torch.softmax(outputs, dim=1)
    valid_mask = (labels != ignore_index)

    top2_vals = torch.topk(probs, k=2, dim=1).values
    p1 = top2_vals[:, 0]
    p2 = top2_vals[:, 1]

    uncertainty = 1.0 - (p1 - p2)
    uncertainty = uncertainty.clamp(min=0.0, max=1.0)

    uncertainty_weight = 1.0 + gamma * uncertainty
    uncertainty_weight = uncertainty_weight.clamp(min=1.0, max=max_weight)
    uncertainty_weight = uncertainty_weight * valid_mask.float()

    return uncertainty_weight, uncertainty


def _safe_valid_ratio(mask, valid_mask, eps=1e-12):
    """Compute mean active-pixel ratio over valid semantic pixels."""
    mask = mask.float()
    valid_mask = valid_mask.float()
    return (mask * valid_mask).sum() / (valid_mask.sum() + eps)

def _masked_weighted_loss(loss_map, weight_map, eps=1e-12):
    weight_sum = weight_map.view(weight_map.shape[0], -1).sum(dim=1)
    weighted_sum = (loss_map * weight_map).view(loss_map.shape[0], -1).sum(dim=1)
    valid_batch = (weight_sum > eps).float()

    if valid_batch.sum() == 0:
        return (loss_map * 0.0).sum()

    sample_loss = weighted_sum / (weight_sum + eps)
    return (sample_loss * valid_batch).sum() / (valid_batch.sum() + eps)


def attack_image(
    model,
    images,
    labels,
    epsilon,
    alpha,
    iterations,
    attack="pgd",
    norm_type="inf",
    num_classes=21,
    out_idx=0,
    debug=False,
    beta=1.0,
    model_arch='deeplabv3',
    use_margin_weight=True,
    use_uncertainty_blind_weight=True,
    high_margin_ratio=0.2,
    uncertainty_gamma=0.5,
    uncertainty_max_weight=1.5,
    uncertainty_blind_lambda=0.3,
    save_steps=None
):
    method_name = attack.lower()
    ce = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=255)

    orig_images = images.detach()
    adv_images = orig_images.clone().detach()

    if norm_type == "inf":
        if method_name == "bim":
            adv_images = adv_images.clone().detach()
        else:
            adv_images = Attack.init_linf(adv_images, epsilon)
    elif norm_type in ["l2", "two"]:
        if method_name == "bim":
            adv_images = adv_images.clone().detach()
        else:
            adv_images = Attack.init_l2(adv_images, epsilon)
        norm_type = "l2"
    else:
        raise ValueError("norm_type must be 'inf', 'l2', or 'two'")

    adv_images = adv_images.detach()
    adv_images.requires_grad_(True)

    attack_logs = []

    return_checkpoints = save_steps is not None
    if save_steps is None:
        save_steps = []
    save_steps = set(int(s) for s in save_steps)
    adv_checkpoints = {}

    for i in range(iterations):
        adv_images = adv_images.detach()
        adv_images.requires_grad_(True)

        outputs = _extract_model_outputs(
            model=model,
            images=adv_images,
            out_idx=out_idx
        )

        ce_loss_pixel = ce(outputs, labels)

        if method_name in ['pgd', 'bim']:
            final_loss = ce_loss_pixel.mean()

        elif method_name == 'segpgd':
            final_loss = Attack.segpgd_scale(
                predictions=outputs,
                labels=labels,
                loss=ce_loss_pixel,
                iteration=i + 1,
                iterations=iterations,
                targeted=False
            )

        elif method_name == 'cospgd':
            scaled_loss = Attack.cospgd_scale(
                predictions=outputs,
                labels=labels,
                loss=ce_loss_pixel,
                num_classes=num_classes,
                targeted=False,
                one_hot=True
            )
            final_loss = scaled_loss.mean()

        elif method_name == 'mr-cospgd':
            # =========================================================
            # Step 1: margin 主攻击
            # =========================================================
            scaled_loss = Attack.cospgd_scale(
                predictions=outputs,
                labels=labels,
                loss=ce_loss_pixel,
                num_classes=num_classes,
                targeted=False,
                one_hot=True
            )

            pred = outputs.argmax(dim=1)
            valid_mask = (labels != 255).float()

            if use_margin_weight:
                raw_margin_weight, margin = _build_margin_weight(
                    outputs=outputs,
                    labels=labels,
                    beta=beta,
                    ignore_index=255
                )
                margin_weight, state_mask, gamma_t = _build_phase_aware_margin_weight(
                    outputs=outputs,
                    labels=labels,
                    raw_margin_weight=raw_margin_weight,
                    iteration=i,
                    iterations=iterations,
                    gamma_min=0.2,
                    ignore_index=255
                )
                high_margin_mask, correct_mask_before = _build_high_margin_mask(
                    margin=margin,
                    labels=labels,
                    pred=pred,
                    top_ratio=high_margin_ratio,
                    ignore_index=255
                )
            else:
                raw_margin_weight = torch.ones_like(ce_loss_pixel)
                margin_weight = torch.ones_like(ce_loss_pixel)
                margin = torch.zeros_like(ce_loss_pixel)
                state_mask = valid_mask
                gamma_t = 1.0
                high_margin_mask = torch.zeros_like(ce_loss_pixel)
                correct_mask_before = ((pred == labels) & (labels != 255)).float()

            base_loss_main = scaled_loss

            # Global margin-aware weighting: all valid pixels are optimized,
            # while margin_weight assigns larger weights to more stable correct pixels.
            main_branch_weight = valid_mask * margin_weight

            main_loss = _masked_weighted_loss(base_loss_main, main_branch_weight)

            main_loss.backward()
            main_grad = adv_images.grad.data

            if norm_type == 'inf':
                adv_images = Attack.step_inf(
                    perturbed_image=adv_images,
                    epsilon=epsilon,
                    data_grad=main_grad,
                    orig_image=orig_images,
                    alpha=alpha,
                    targeted=False
                )
            elif norm_type == 'l2':
                adv_images = Attack.step_l2(
                    perturbed_image=adv_images,
                    epsilon=epsilon,
                    data_grad=main_grad,
                    orig_image=orig_images,
                    alpha=alpha,
                    targeted=False
                )
            else:
                raise ValueError("norm_type must be 'inf', 'l2', or 'two'")

            # =========================================================
            # Step 2: blind 补刀攻击（基于更新后的 adv_images）
            # =========================================================
            adv_images = adv_images.detach()
            adv_images.requires_grad_(True)

            outputs_blind = _extract_model_outputs(
                model=model,
                images=adv_images,
                out_idx=out_idx
            )

            ce_loss_pixel_blind = ce(outputs_blind, labels)

            scaled_loss_blind = Attack.cospgd_scale(
                predictions=outputs_blind,
                labels=labels,
                loss=ce_loss_pixel_blind,
                num_classes=num_classes,
                targeted=False,
                one_hot=True
            )

            pred_blind = outputs_blind.argmax(dim=1)
            correct_mask_after = ((pred_blind == labels) & (labels != 255)).float()

            # Recompute current high-margin region after the main update.
            # Final setting: residual Blind attacks currently correct pixels
            # excluding the current high-margin region.
            raw_margin_weight_blind, margin_blind = _build_margin_weight(
                outputs=outputs_blind,
                labels=labels,
                beta=beta,
                ignore_index=255
            )
            margin_weight_blind, state_mask_blind, gamma_t_blind = _build_phase_aware_margin_weight(
                outputs=outputs_blind,
                labels=labels,
                raw_margin_weight=raw_margin_weight_blind,
                iteration=i,
                iterations=iterations,
                gamma_min=0.2,
                ignore_index=255
            )
            high_margin_mask_blind, correct_mask_before_blind = _build_high_margin_mask(
                margin=margin_blind,
                labels=labels,
                pred=pred_blind,
                top_ratio=high_margin_ratio,
                ignore_index=255
            )

            blind_mask = correct_mask_after * (1.0 - high_margin_mask_blind)

            if use_uncertainty_blind_weight:
                uncertainty_weight, uncertainty_map = _build_uncertainty_weight(
                    outputs=outputs_blind,
                    labels=labels,
                    gamma=uncertainty_gamma,
                    max_weight=uncertainty_max_weight,
                    ignore_index=255
                )
            else:
                uncertainty_weight = torch.ones_like(ce_loss_pixel_blind)
                uncertainty_map = torch.zeros_like(ce_loss_pixel_blind)

            base_loss_blind = scaled_loss_blind
            blind_branch_weight = blind_mask * uncertainty_weight

            if use_uncertainty_blind_weight:
                blind_loss = _masked_weighted_loss(base_loss_blind, blind_branch_weight)

                if blind_loss.requires_grad and blind_branch_weight.detach().sum() > 0:
                    blind_objective = uncertainty_blind_lambda * blind_loss
                    blind_objective.backward()

                    blind_grad = adv_images.grad.data

                    if norm_type == 'inf':
                        adv_images = Attack.step_inf(
                            perturbed_image=adv_images,
                            epsilon=epsilon,
                            data_grad=blind_grad,
                            orig_image=orig_images,
                            alpha=alpha,
                            targeted=False
                        )
                    elif norm_type == 'l2':
                        adv_images = Attack.step_l2(
                            perturbed_image=adv_images,
                            epsilon=epsilon,
                            data_grad=blind_grad,
                            orig_image=orig_images,
                            alpha=alpha,
                            targeted=False
                        )
                    else:
                        raise ValueError("norm_type must be 'inf', 'l2', or 'two'")
            else:
                blind_loss = base_loss_blind.new_zeros(())

            final_loss = main_loss + uncertainty_blind_lambda * blind_loss
            combined_weight = main_branch_weight + uncertainty_blind_lambda * blind_branch_weight

        else:
            raise ValueError(f"Unknown method: {method_name}")

        if method_name != 'mr-cospgd':
            final_loss.backward()
            update_grad = adv_images.grad.data

            if norm_type == 'inf':
                adv_images = Attack.step_inf(
                    perturbed_image=adv_images,
                    epsilon=epsilon,
                    data_grad=update_grad,
                    orig_image=orig_images,
                    alpha=alpha,
                    targeted=False
                )
            elif norm_type == 'l2':
                adv_images = Attack.step_l2(
                    perturbed_image=adv_images,
                    epsilon=epsilon,
                    data_grad=update_grad,
                    orig_image=orig_images,
                    alpha=alpha,
                    targeted=False
                )
            else:
                raise ValueError("norm_type must be 'inf', 'l2', or 'two'")

        if method_name == 'mr-cospgd':
            attack_logs.append({
                'iter': i + 1,
                'final_loss': final_loss.item(),
                'main_loss': main_loss.item(),
                'blind_loss': blind_loss.item() if isinstance(blind_loss, torch.Tensor) else float(blind_loss),
                'high_margin_ratio_mean': high_margin_mask.mean().item(),
                'blind_ratio_mean': blind_mask.mean().item(),
                # GM-BRC synergy statistics, computed over valid semantic pixels.
                'correct_ratio_before_gm': _safe_valid_ratio(correct_mask_before, valid_mask).item(),
                'correct_ratio_after_gm': _safe_valid_ratio(correct_mask_after, valid_mask).item(),
                'residual_blind_ratio': _safe_valid_ratio(blind_mask, valid_mask).item(),
                'margin_weight_mean': margin_weight.mean().item(),
                'margin_weight_max': margin_weight.max().item(),
                'uncertainty_weight_mean': uncertainty_weight.mean().item(),
                'uncertainty_weight_max': uncertainty_weight.max().item(),
                'gamma_t': gamma_t,
                'uncertainty_blind_lambda': uncertainty_blind_lambda,
                'blind_region_mode': 'residual'
            })
        else:
            attack_logs.append({
                'iter': i + 1,
                'final_loss': final_loss.item()
            })

        current_step = i + 1
        if current_step in save_steps:
            adv_checkpoints[current_step] = adv_images.detach().clone()

        adv_images = adv_images.detach()
        adv_images.requires_grad_(True)

    if return_checkpoints:
        return adv_images.detach(), attack_logs, adv_checkpoints
    return adv_images.detach(), attack_logs


def get_attack(
    model,
    images,
    labels,
    device,
    method_name='cospgd',
    epsilon=8 / 255,
    alpha=0.01,
    iterations=40,
    num_classes=21,
    beta=1.0,
    norm_type='inf',
    debug=False,
    out_idx=0,
    model_arch='deeplabv3',
    use_margin_weight=True,
    use_uncertainty_blind_weight=True,
    high_margin_ratio=0.4,
    uncertainty_gamma=0.5,
    uncertainty_max_weight=1.5,
    uncertainty_blind_lambda=0.1,
    save_steps=None
):
    images = images.to(device)
    labels = labels.to(device)

    attack_result = attack_image(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        attack=method_name,
        norm_type=norm_type,
        num_classes=num_classes,
        out_idx=out_idx,
        debug=debug,
        beta=beta,
        model_arch=model_arch,
        use_margin_weight=use_margin_weight,
        use_uncertainty_blind_weight=use_uncertainty_blind_weight,
        high_margin_ratio=high_margin_ratio,
        uncertainty_gamma=uncertainty_gamma,
        uncertainty_max_weight=uncertainty_max_weight,
        uncertainty_blind_lambda=uncertainty_blind_lambda,
        save_steps=save_steps
    )

    return attack_result
