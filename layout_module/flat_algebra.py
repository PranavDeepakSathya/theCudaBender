from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from .composition import Flat_tuple_morphism

from .aster import Pointed
  

def get_col_major_stride(shape: Tuple[int]): 
  col_major = [1]
  for i in range(1, len(shape)): 
    col_major.append(col_major[i-1]*shape[i-1])
  return tuple(col_major)

def get_row_major_stride(shape: Tuple[int]): 
  row_major = [1]*len(shape)
  for i in reversed(range(len(shape)-1)): 
    row_major[i] = row_major[i+1]*shape[i+1] 
  return tuple(row_major)
  
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


def lex(shape: Tuple[int], co_ordinates: Tuple[int]): 
  is_flat_compatible(shape, co_ordinates)
  rs_stride = get_row_major_stride(shape)
  res = 0
  for i in range(len(shape)): 
    res += rs_stride[i]*co_ordinates[i]
    
  return res 

def lex_inv(shape: Tuple[int], idx: int): 
  size = reduce(mul,shape,1)
  assert 0 <= idx < size 
  rs_stride = get_row_major_stride(shape) 
  res = []
  for i in range(len(shape)): 
    res.append((idx//rs_stride[i])%shape[i])
    
  return tuple(res)
    
def coordinate_map(stride: Tuple[int], co_ordinates: Tuple[int]): 
  assert len(stride) == len(co_ordinates)
  res = 0
  for i in range(len(stride)): 
    res += stride[i]*co_ordinates[i]
    
  return res
    
def layout_map(shape:Tuple[int], stride:Tuple[int], idx: int): 
  assert len(shape) == len(stride)
  size = reduce(mul,shape,1)
  assert 0 <= idx < size 
  return coordinate_map(stride, colex_inv(shape, idx))

def sort(shape: Tuple[int], stride: Tuple[int]): 
  assert len(shape) == len(stride)
  m = len(shape)
  shape = list(shape)
  stride = list(stride)
  for i in range(m): 
    for j in range(i+1, m):
      if stride[i] > stride[j]: 
        stride[i], stride[j] = stride[j], stride[i]
        shape[i], shape[j] = shape[j], shape[i]
        
        
  for i in range(m-1): 
    if stride[i] == stride[i+1] and shape[i] > shape[i+1]: 
      shape[i],shape[i+1] = shape[i+1],shape[i]
      
  return tuple(shape), tuple(stride)

def squeeze(shape: Tuple[int], stride: Tuple[int]): 
  assert len(shape) == len(stride)
  new_shape = []
  new_stride = []
  for i in range(len(shape)): 
    if shape[i] > 1: 
      new_shape.append(shape[i])
      new_stride.append(stride[i])
      
  return tuple(new_shape), tuple(new_stride)

def filter_zeros(shape: Tuple[int], stride: Tuple[int]): 
  assert len(shape) == len(stride)
  new_shape = []
  new_stride = []
  for i in range(len(stride)): 
    if stride[i] > 0: 
      new_shape.append(shape[i])
      new_stride.append(stride[i])
      
  return tuple(new_shape), tuple(new_stride)  
  
def divides (a:Tuple[int], b:Tuple[int]) -> bool: 
  if len(a) <= len(b): 
    for i in range(len(a)): 
      if a[i] != b[i]: 
        return False 
    return True 
  return False
      
      
def get_quotient(a:Tuple[int], b:Tuple[int]) -> bool: 
  assert divides(a,b) == True 
  start = a.length 
  return b[start:]
  
def is_flat_tractable(shape:Tuple[int], stride:Tuple[int]) -> bool: 
  assert len(shape) == len(stride)
  sf, df = filter_zeros(shape, stride)
  sh, dh = sort(sf,df)
  s,d = squeeze(sh, dh)
  for i in range(len(s)-1): 
    if d[i+1] % (s[i]*d[i]) != 0: 
      return False
  return True 


def flat_layout_to_mor(shape, stride):
    assert len(shape) == len(stride)
    assert is_flat_tractable(shape, stride)
    axes = [
        (i, shape[i], stride[i])
        for i in range(len(shape))
        if stride[i] != 0
    ]

    axes.sort(key=lambda x: x[2])

    map = [Pointed.astr] * len(shape)
    co_domain = []

    prev_prod = 1

    for i, s, d in axes:
        if d == prev_prod:
            # append (s)
            co_domain.append(s)
            map[i] = len(co_domain) - 1
            prev_prod *= s
        else:
            # append (d/prev_prod, s)
            co_domain.append(d // prev_prod)
            co_domain.append(s)
            map[i] = len(co_domain) - 1   # map to the s_j slot
            prev_prod = d * s

    return Flat_tuple_morphism(shape, tuple(co_domain), tuple(map))

      

    
    