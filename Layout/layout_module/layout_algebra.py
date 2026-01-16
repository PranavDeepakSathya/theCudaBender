
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
from .layout import Layout
from .composition import nested_tuple_morphism, just_compose, compose_morphism




def concatenate(ls:List[Layout])-> Layout: 
  shapes = [l.shape for l in ls]
  strides = [l.stride for l in ls]
  cat_shape = na.concatenate(shapes)
  cat_stride = na.concatenate(strides)
  return Layout(cat_shape, cat_stride)

def substitute_modes(L:Layout, P:Profile)-> Layout: 
  assert L.rank == P.length 
  shape_modes = [L.get_mode(i).shape for i in range(L.rank)]
  stride_modes = [L.get_mode(i).stride for i in range(L.rank)]
  new_shape = na.substitute(P, shape_modes)
  new_stride = na.substitute(P, stride_modes)
  return Layout(new_shape, new_stride)

def get_refine_relative_mode(layout:Layout, coal_shape: NestedTuple,i:int) -> "Layout": 
  assert isinstance(coal_shape, NestedTuple)
  assert 0 <= i < coal_shape.length 
  assert na.refines(layout.shape, coal_shape) == True 
  if coal_shape.depth == 0: 
    return layout
  j = 0 
  while coal_shape.get_prefix_length(j+1) <= i: 
    j+=1 
  N = coal_shape.get_prefix_length(j) 
  return get_refine_relative_mode(layout.get_mode(j), coal_shape.get_mode(j), i-N)
  
def construct_morphism(A:Layout)->nested_tuple_morphism: 
  assert A.is_tractable()
  flat_mor = fa.flat_layout_to_mor(A.shape.int_tuple, A.stride.int_tuple)
  return nested_tuple_morphism(A.shape, NestedTuple.from_literal(flat_mor.co_domain), flat_mor.map)

def compose (A:Layout, B:Layout):
  A_mor,B_mor = construct_morphism(A), construct_morphism(B)
  return compose_morphism(A_mor,B_mor)

def divide (A:Layout, B:Layout):
  #A_mor,B_mor = construct_morphism(A), construct_morphism(B)
  #assert B_mor.co_domain == A_mor.domain
  B_comp = B.construct_N_complement(A.size)
  cat = concatenate([B,B_comp])
  return compose(cat, A)
