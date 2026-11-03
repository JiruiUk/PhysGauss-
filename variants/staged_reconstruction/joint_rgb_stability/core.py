"""A single declared common epsilon change, with no hidden rate/moment changes."""


def condition_adam(optimizers, enabled=False):
    if enabled:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['eps'] = 1e-8
    return optimizers
