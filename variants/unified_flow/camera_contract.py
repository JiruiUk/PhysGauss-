"""Match the actual released loader convention, including its FoV assignment.

Existing reconstruction weights were trained with FovX=fovy, FovY=fovx.
Changing the original loader would invalidate their camera/geometry contract.
"""
import math


def loader_fovs(metadata_fovx, width=1600, height=900):
    vertical = 2*math.atan(math.tan(metadata_fovx/2)*height/width)
    return vertical, metadata_fovx
