from Dassl.dassl.utils import Registry, check_availability
from trainers.CLIP import CLIP
from trainers.BLEND import BLEND

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(BLEND)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
