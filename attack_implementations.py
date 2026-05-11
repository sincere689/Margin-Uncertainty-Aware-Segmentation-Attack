import torch


class Attack:
    def __init__(self):
        pass

    @staticmethod
    def step_inf(
        perturbed_image,
        epsilon,
        data_grad,
        orig_image,
        alpha,
        targeted,
        clamp_min=0,
        clamp_max=1,
        grad_scale=None
    ):
        sign_data_grad = alpha * data_grad.sign()

        if targeted:
            sign_data_grad *= -1
        if grad_scale is not None:
            sign_data_grad *= grad_scale

        perturbed_image = perturbed_image.detach() + sign_data_grad
        delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image

    @staticmethod
    def step_l2(
        perturbed_image,
        epsilon,
        data_grad,
        orig_image,
        alpha,
        targeted,
        clamp_min=0,
        clamp_max=1,
        grad_scale=None
    ):
        if targeted:
            data_grad *= -1

        if grad_scale is not None:
            data_grad *= grad_scale

        perturbed_image = perturbed_image.detach() + alpha * data_grad
        delta = Attack.lp_normalize(
            noise=perturbed_image - orig_image,
            p=2,
            epsilon=epsilon,
            decrease_only=True
        )
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image

    @staticmethod
    def lp_normalize(noise, p, epsilon=None, decrease_only=False):
        if epsilon is None:
            epsilon = torch.tensor(1.0, device=noise.device, dtype=noise.dtype)

        denom = torch.norm(noise, p=p, dim=(-1, -2, -3))
        denom = torch.maximum(
            denom,
            torch.tensor(1e-12, device=noise.device, dtype=noise.dtype)
        ).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        if decrease_only:
            denom = torch.maximum(
                denom / epsilon,
                torch.tensor(1.0, device=noise.device, dtype=noise.dtype)
            )
        else:
            denom = denom / epsilon

        return noise / denom

    @staticmethod
    def final_project(
        perturbed_image,
        orig_image,
        epsilon,
        norm_type='inf',
        clamp_min=0,
        clamp_max=1
    ):
        if norm_type == 'two':
            norm_type = 'l2'

        delta = perturbed_image - orig_image
        if norm_type == 'inf':
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        elif norm_type == 'l2':
            delta = Attack.lp_normalize(
                noise=delta,
                p=2,
                epsilon=epsilon,
                decrease_only=True
            )
        else:
            raise ValueError("norm_type must be 'inf', 'l2', or 'two'")

        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image

    @staticmethod
    def init_linf(images, epsilon, clamp_min=0, clamp_max=1):
        noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images

    @staticmethod
    def init_l2(images, epsilon, clamp_min=0, clamp_max=1):
        noise = torch.empty_like(images).uniform_(-1, 1)
        noise = Attack.lp_normalize(
            noise=noise,
            p=2,
            epsilon=epsilon,
            decrease_only=False
        )
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images

    @staticmethod
    def _normalize_l2_grad(grad):
        grad_flat = grad.reshape(grad.shape[0], -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1)
        grad_norm = torch.clamp(grad_norm, min=1e-12).reshape(-1, 1, 1, 1)
        return grad / grad_norm

    @staticmethod
    def segpgd_scale(
        predictions,
        labels,
        loss,
        iteration,
        iterations,
        targeted=False,
    ):
        lambda_t = iteration / (2 * iterations)
        output_idx = torch.argmax(predictions, dim=1)

        if targeted:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    lambda_t * loss,
                    (1 - lambda_t) * loss
                )
            ) / (predictions.shape[-2] * predictions.shape[-1])
        else:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    (1 - lambda_t) * loss,
                    lambda_t * loss
                )
            ) / (predictions.shape[-2] * predictions.shape[-1])

        return loss

    @staticmethod
    def cospgd_scale(
        predictions,
        labels,
        loss,
        num_classes=None,
        targeted=False,
        one_hot=True,
    ):
        if one_hot:
            transformed_target = torch.nn.functional.one_hot(
                torch.clamp(labels, labels.min(), num_classes - 1),
                num_classes=num_classes
            ).permute(0, 3, 1, 2)
        else:
            transformed_target = torch.nn.functional.softmax(labels, dim=1)

        cossim = torch.nn.functional.cosine_similarity(
            torch.nn.functional.softmax(predictions, dim=1),
            transformed_target,
            dim=1
        )

        if targeted:
            cossim = 1 - cossim

        loss = cossim.detach() * loss
        return loss