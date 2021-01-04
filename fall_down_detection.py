import numpy as np


def fall(pose):

    upper_limb = [0, 1, 2, 3, 4, 5]
    lower_limb = [10, 11, 12, 13, 14, 15]
    limb = upper_limb + lower_limb

    pose_limb = pose[limb, 1]
    pose_upper_limb = pose[upper_limb, 1]
    pose_lower_limb = pose[lower_limb, 1]

    limb_range = max(pose_limb) - min(pose_limb)
    coincide = (max(pose_lower_limb) - min(pose_upper_limb))/limb_range
    if coincide > 0.3:
        return 1
    return 0



