from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul

def get_col_major_stride(shape: Tuple[int]): 
  col_major = [1]
  for i in range(1, len(shape)): 
    col_major.append(col_major[i-1]*shape[i-1])
  return tuple(col_major)

def is_flat_compatible(shape: Tuple[int], co_ordinates: Tuple[int]): 
  assert len(shape) == len(co_ordinates)
  for i in range(len(shape)): 
    assert 0 <= co_ordinates[i] < shape[i]
    
def colex(shape: Tuple[int], co_ordinates: Tuple[int]): 
  is_flat_compatible(shape, co_ordinates)
  cs_stride = get_col_major_stride(shape)
  res = 0
  for i in range(len(shape)): 
    res += cs_stride[i]*co_ordinates[i]
    
  return res 

def colex_inv(shape: Tuple[int], idx:int): 
  size = reduce(mul,shape,1)
  assert 0 <= idx < size 
  cs_stride = get_col_major_stride(shape) 
  res = []
  for i in range(len(shape)): 
    res.append((idx//cs_stride[i])%shape[i])
    
  return tuple(res)
    
def coordinate_map(stride: Tuple[int], co_ordinates: Tuple[int]): 
  assert len(stride) == len(co_ordinates)
  res = 0
  for i in range(len(stride)): 
    res += stride[i]*co_ordinates[i]
    
  return res
    
def layout_map(shape:Tuple[int], stride:Tuple[int], idx: int): 
  size = reduce(mul,shape,1)
  assert 0 <= idx < size 
  return coordinate_map(stride, colex_inv(shape, idx))