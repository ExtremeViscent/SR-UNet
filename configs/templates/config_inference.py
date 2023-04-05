from colossalai.amp import AMP_TYPE
import os

fp16=dict(
    mode=AMP_TYPE.TORCH
)

parallel = dict(
    data=dict(size=1),
)