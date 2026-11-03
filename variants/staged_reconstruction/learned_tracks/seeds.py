"""Deterministic dense texture proposals; acceptance thresholds are unchanged."""
import cv2
import numpy as np


def proposals(gray, mask, maximum=64, distance=4., minimum_std=2.):
    gray=np.asarray(gray,np.float32)
    mean=cv2.blur(gray,(5,5))
    std=np.sqrt(np.maximum(cv2.blur(gray*gray,(5,5))-mean*mean,0))
    ys,xs=np.nonzero(np.asarray(mask,bool)&(std>=minimum_std))
    order=np.lexsort((xs,ys,-std[ys,xs]))
    selected=[]
    for index in order:
        xy=np.array([xs[index],ys[index]],np.float32)
        if selected and np.linalg.norm(np.asarray(selected)-xy,axis=-1).min()<distance: continue
        selected.append(xy)
        if len(selected)>=maximum: break
    return np.asarray(selected,np.float32).reshape(-1,2)
