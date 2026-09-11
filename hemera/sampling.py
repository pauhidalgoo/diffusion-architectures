from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .interfaces import Denoiser, TextCondition
from .objectives import GenerativeObjective


@torch.no_grad()
def sample_latents(
    model: Denoiser,
    objective: GenerativeObjective,
    shape: tuple[int, ...],
    condition: TextCondition,
    steps: int = 30,
    guidance_scale: float = 3.0,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    method: str = "euler",
    unconditional_condition: TextCondition | None = None,
) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = condition.hidden_states.device
    dtype = condition.hidden_states.dtype
    if isinstance(generator, Sequence):
        if len(generator) != shape[0]:
            raise ValueError(
                "A generator sequence must contain one generator per sample"
            )
        x = torch.cat(
            [
                torch.randn(
                    (1, *shape[1:]),
                    device=device,
                    dtype=dtype,
                    generator=item,
                )
                for item in generator
            ],
            dim=0,
        )
    else:
        x = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
    unconditional = unconditional_condition or TextCondition.unconditional_like(condition)
    times = objective.sampling_times(steps, device)

    def guided_velocity(state: Tensor, time_value: Tensor) -> Tensor:
        batch_time = time_value.expand(state.shape[0])
        if guidance_scale == 1.0:
            output = model(state, batch_time, condition)
        else:
            combined_condition = TextCondition(
                hidden_states=torch.cat(
                    [unconditional.hidden_states, condition.hidden_states], dim=0
                ),
                attention_mask=torch.cat(
                    [unconditional.attention_mask, condition.attention_mask], dim=0
                ),
                pooled=(
                    None
                    if condition.pooled is None or unconditional.pooled is None
                    else torch.cat([unconditional.pooled, condition.pooled], dim=0)
                ),
            )
            combined_output = model(
                torch.cat([state, state], dim=0),
                torch.cat([batch_time, batch_time], dim=0),
                combined_condition,
            )
            unconditional_output, conditional_output = combined_output.chunk(2)
            output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
        return objective.velocity(output, state, batch_time)

    for index in range(steps):
        current, following = times[index], times[index + 1]
        dt = following - current
        velocity = guided_velocity(x, current)
        if method == "heun" and index < steps - 1:
            proposal = x + dt * velocity
            corrected = guided_velocity(proposal, following)
            x = x + dt * 0.5 * (velocity + corrected)
        else:
            x = x + dt * velocity
    return x
