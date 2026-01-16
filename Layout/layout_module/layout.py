

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from .nested_tuple import NestedTuple
from . import layout_algebra as la 
from .profile import Profile,Atom
from . import nested_tuple_algebra as na
from .import flat_algebra as fa 


@dataclass(frozen=True)
class Layout:
  shape: NestedTuple
  stride: NestedTuple

  def __post_init__(self):
    assert isinstance(self.shape, NestedTuple)
    assert isinstance(self.stride, NestedTuple)
    assert self.shape.prof == self.stride.prof
    
    
  def get_mode(self, i:int)->"Layout": 
    assert 0 <= i < self.rank 
    return Layout(self.shape.get_mode(i), self.stride.get_mode(i))
  
  def get_entry(self, i:int)->"Layout": 
    assert 0 <= i < self.length 
    return Layout(self.shape.get_entry(i), self.stride.get_entry(i))
  
  def is_flat(self)->bool: 
    return self.shape.is_flat() and self.stride.is_flat()
    
  def flatten(self)->"Layout": 
    return Layout(self.shape.flatten(), self.stride.flatten())
  
  
 
  def coalesce(self) -> "Layout":
    Lf = self.flatten()

    shape = Lf.shape.int_tuple
    stride = Lf.stride.int_tuple
    m = len(shape)

    # Empty layout: do nothing
    if m == 0:
      return Lf

    new_shape = []
    new_stride = []

    curr_shape = shape[0]
    curr_stride = stride[0]

    for i in range(1, m):
      if stride[i] == curr_shape * curr_stride:
        curr_shape *= shape[i]
      else:
        new_shape.append(curr_shape)
        new_stride.append(curr_stride)
        curr_shape = shape[i]
        curr_stride = stride[i]

    new_shape.append(curr_shape)
    new_stride.append(curr_stride)

    # Build flat profile of resulting rank
    flat_prof = Profile(tuple(Profile(Atom.STAR) for _ in new_shape))

    return Layout(
      NestedTuple(tuple(new_shape), flat_prof),
      NestedTuple(tuple(new_stride), flat_prof),
    )
    
  def relative_coalesce(self, coal_shape:NestedTuple)-> "Layout": 
    LS = [la.get_refine_relative_mode(self,coal_shape,i).coalesce() for i in range(coal_shape.length)]
    return la.concatenate(LS)

  def is_compact(self)-> bool: 
    s = self.shape.int_tuple
    d = self.stride.int_tuple 
    sp, dp = fa.squeeze(s,d)
    s_, d_ = fa.sort(sp,dp)
    s_cs = fa.get_col_major_stride(s_)
    if d_ == s_cs:
      return True 
    return False
  
  def is_complementable(self,N:int)->bool: 
    s,d = self.shape.int_tuple, self.stride.int_tuple
    sp,dp = fa.squeeze(s,d)
    s_, d_, = fa.sort(sp,dp)
    for i in range(0, len(s_)-1): 
      if d_[i+1] % (s_[i]*d_[i]) != 0:
        return (False,False)
    
    if (N % (s_[-1]*d_[-1]) == 0): 
      return (True, True)
    else: 
      return (True, False)
    
  def is_tractable(self)-> bool: 
    s,d = self.shape.int_tuple, self.stride.int_tuple
    sf,df = fa.filter_zeros(s,d)
    sp,dp = fa.squeeze(sf,df)
    s_, d_ = fa.sort(sp,dp)
    for i in range(0,len(s_)-1): 
      if d_[i+1] % (s_[i]*d_[i]) != 0: 
        return False 
    
    return True
  
  def construct_complement(self)->"Layout": 
    assert self.is_complementable(0) == (True,True) 
    s,d = self.shape.int_tuple, self.stride.int_tuple
    sp,dp = fa.squeeze(s,d)
    t,e = fa.sort(sp,dp)
    C_shape = [e[0]]
    C_stride = [1]
    for i in range(1, len(t)): 
      C_stride.append(t[i-1]*e[i-1])
      C_shape.append(e[i]//(t[i-1]*e[i-1]))
    flat_profile = Profile(tuple([Profile(Atom.STAR) for _ in range(len(C_stride))]))
    C_shape_nest = NestedTuple(tuple(C_shape), flat_profile)
    C_stride_nest = NestedTuple(tuple(C_stride), flat_profile)
    C_layout = Layout(C_shape_nest, C_stride_nest)
    comp = C_layout.coalesce()
    return comp
  
  def construct_N_complement(self, N)->"Layout": 
    assert self.is_complementable(N) == (True,True) 
    s,d = self.shape.int_tuple, self.stride.int_tuple
    sp,dp = fa.squeeze(s,d)
    t,e = fa.sort(sp,dp)
    C_shape = [e[0]]
    C_stride = [1]
    for i in range(1, len(t)): 
      C_stride.append(t[i-1]*e[i-1])
      C_shape.append(e[i]//(t[i-1]*e[i-1]))
      
    C_stride.append(t[-1]*e[-1])
    C_shape.append((N//(t[-1]*e[-1])))
    flat_profile = Profile(tuple([Profile(Atom.STAR) for _ in range(len(C_stride))]))
    C_shape_nest = NestedTuple(tuple(C_shape), flat_profile)
    C_stride_nest = NestedTuple(tuple(C_stride), flat_profile)
    C_layout = Layout(C_shape_nest, C_stride_nest)
    comp = C_layout.coalesce()
    return comp 
  

  
    
  
  @property
  def rank(self) -> int:
    return self.shape.rank

  @property
  def length(self) -> int:
    return self.shape.length

  @property
  def depth(self) -> int:
    return self.shape.depth

  @property
  def size(self) -> int:
    return self.shape.size

  @property
  def cosize(self) -> int:
    shape = self.shape.int_tuple
    stride = self.stride.int_tuple
    assert len(shape) == len(stride)

    if len(shape) == 0:
      return 0

    return 1 + sum((s - 1) * st for s, st in zip(shape, stride))


    
    