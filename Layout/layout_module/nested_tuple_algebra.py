from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from .nested_tuple import NestedTuple
from .profile import Profile, Atom 
from . import flat_algebra as fa


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




def mutual_refinement(T: NestedTuple, U: NestedTuple):
  X = list(T.int_tuple)
  Y = list(U.int_tuple)

  Xp, Yp = (), ()
  X_mode, Y_mode = (), ()
  i,j = 0,0

  while i < len(X) and j < len(Y):
    xi, yj = X[i], Y[j]

    if xi == yj:
      X_mode += (xi,)
      Xp += (X_mode,)
      X_mode = ()

      Y_mode += (yj,)
      Yp += (Y_mode,)
      Y_mode = ()

      i += 1
      j += 1

    elif yj % xi == 0:
      X_mode += (xi,)
      Xp += (X_mode,)
      X_mode = ()

      Y_mode += (xi,)
      Y[j] = yj // xi
      i += 1

    elif xi % yj == 0:
      Y_mode += (yj,)
      Yp += (Y_mode,)
      Y_mode = ()

      X_mode += (yj,)
      X[i] = xi // yj
      j += 1

    else:
      return None

  
  if Y_mode:
    Y_mode += (Y[j],)
    Yp += (Y_mode,)
    j+= 1

  while j < len(Y):
    Yp += (Y[j],)
    j += 1
      
  #return Xp, Yp
  X_refined = NestedTuple.from_literal(Xp)
  Y_refined = NestedTuple.from_literal(Yp)
  T_prof = T.prof
  U_prof = U.prof 
  X_modes = [X_refined.get_mode(i) for i in range(X_refined.rank)]
  Y_modes = [Y_refined.get_mode(i) for i in range(Y_refined.rank)]
  X_fin = substitute(T_prof, X_modes)
  Y_fin = substitute(U_prof, Y_modes)
  return X_fin, Y_fin

      