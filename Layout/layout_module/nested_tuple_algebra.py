from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from .nested_tuple import NestedTuple
from .profile import Profile, Atom 

def refines(a:NestedTuple,b:NestedTuple):
  assert isinstance(a, NestedTuple) and isinstance(b, NestedTuple)
  
  if b.rank == 1 and b.prof.is_atom(): 
    return a.size == b.size 
  
  if a.depth > 0 and b.depth > 0: 
    if a.rank != b.rank: 
      return False 
    for i in range(a.rank): 
      if not refines(a.get_mode(i), b.get_mode(i)): 
        return False 
    return True 
  return False
    
def get_refine_relative_mode(a:NestedTuple, b:NestedTuple, i:int): 
  assert isinstance(a, NestedTuple) and isinstance(b, NestedTuple)
  assert 0 <= i < b.length
  assert refines(a,b) == True 
  if b.depth == 0: 
    return a
  j = 0 
  
  while b.get_prefix_length(j+1) <= i: 
    j+=1 
  N = b.get_prefix_length(j) 
  
  return get_refine_relative_mode(a.get_mode(j), b.get_mode(j), i-N)
  
   
def get_refine_relative_flatten(a:NestedTuple, b:NestedTuple): 
  assert isinstance(a, NestedTuple) and isinstance(b, NestedTuple)
  assert refines(a,b) == True 
  rel_modes = [get_refine_relative_mode(a,b,i) for i in range(b.length)]
  
  return concatenate(rel_modes)
    
       
  
def concatenate(nts: List[NestedTuple]): 
  flat_tuple = ()
  flat_profile = []
  for nt in nts: 
    flat_tuple += nt.int_tuple
    flat_profile.append(nt.prof)
    
  return NestedTuple(flat_tuple, Profile(tuple(flat_profile)))



def substitute(
  prof: Profile,
  nts: list["NestedTuple"]
) -> "NestedTuple":
  assert prof.length == len(nts)

  int_tuple = ()
  profiles = []

  for nt in nts:
    assert isinstance(nt, NestedTuple)
    assert isinstance(nt.prof, Profile)
    int_tuple += nt.int_tuple
    profiles.append(nt.prof)

  new_prof = prof.substitute(profiles)
  return NestedTuple(int_tuple, new_prof)