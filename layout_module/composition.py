from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from collections import defaultdict
from .aster import Pointed
from .nested_tuple import NestedTuple
from . import nested_tuple_algebra as na
import matplotlib.pyplot as plt


class Flat_tuple_morphism: 
  def __init__ (self, domain:Tuple[int], co_domain:Tuple[int], map: Tuple[int|Pointed]): 
    self.m = len(domain)
    self.n = len(co_domain)
    assert len(map) == self.m 
    pre_img_dict = defaultdict(list) 
    for i in range(self.m): 
      val = map[i]
      if val != Pointed.astr: 
        assert 0 <= val < self.n 
        assert domain[i] == co_domain[map[i]]
        pre_img_dict[val].append(i)
        
    for i in range(self.n): 
      assert len(pre_img_dict[i]) <= 1
      
    self.map = map
    self.domain = domain 
    self.co_domain = co_domain
    self.preimg = pre_img_dict
      
  def __repr__(self):
    return (
        f"flat_tuple_morphism("
        f"domain={self.domain}, "
        f"co_domain={self.co_domain}, "
        f"map={self.map}"
        f")"
    )


  def draw(self):
    import matplotlib.pyplot as plt

    S = self.domain
    T = self.co_domain
    a = self.map

    m = len(S)
    n = len(T)

    fig, ax = plt.subplots(figsize=(3, max(m, n) * 0.6))

    xS = 0
    xT = 2

    # y positions: bottom → top corresponds to index order
    yS = list(range(m))
    yT = list(range(n))

    # draw domain values
    for i, y in enumerate(yS):
        ax.text(xS, y, str(S[i]), ha="center", va="center")

    # draw codomain values
    for j, y in enumerate(yT):
        ax.text(xT, y, str(T[j]), ha="center", va="center")

    # draw arrows
    pad = 0.25
    for i, v in enumerate(a):
        if v != Pointed.astr:
            ax.annotate(
                "",
                xy=(xT - pad, yT[v]),
                xytext=(xS + pad, yS[i]),
                arrowprops=dict(arrowstyle="->", linewidth=1.5),
            )

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, max(m, n) - 0.5)
    ax.axis("off")
    plt.show()



class nested_tuple_morphism: 
  def __init__ (self, domain:NestedTuple, co_domain:NestedTuple, map:Tuple[int|Pointed]): 
    self.morphism = Flat_tuple_morphism(domain.int_tuple, co_domain.int_tuple, map)
    self.domain = domain 
    self.co_domain = co_domain
    
    
  def get_entry(self, i): 
    assert 0 <= i <= self.domain.length
    new_map = (self.morphism.map[i],)
    new_domain = self.domain.get_entry(i)
    new_co_domain = self.co_domain
    return nested_tuple_morphism(new_domain, new_co_domain, new_map)
  
  def get_mode(self,i): 
    assert 0 <= i <= self.domain.rank 
    N = self.domain.get_prefix_length(i)
    new_domain = self.domain.get_mode(i)
    new_co_domain = self.co_domain
    l = new_domain.length
    new_map = tuple([self.morphism.map[N+i] for i in range(l)])
    return nested_tuple_morphism(new_domain, new_co_domain, new_map)
  
  
  def make_pullback(self, refined_co_domain:NestedTuple): 
    assert na.refines(refined_co_domain, self.co_domain) == True 
    refined_domain_candidate = []
    for i in range(self.domain.length): 
      if self.morphism.map[i] == Pointed.astr: 
        refined_domain_candidate.append(self.domain.get_entry(i))
      else: 
        j = self.morphism.map[i]
        refined_domain_candidate.append(na.get_refine_relative_mode(refined_co_domain,self.co_domain,j))
        
    refined_domain = na.substitute(self.domain.prof, refined_domain_candidate)
    refined_morphism = []
    refined_co_domain_rel_flat = na.get_refine_relative_flatten(refined_co_domain, self.co_domain)
    for i in range(len(refined_domain_candidate)): #relative modes
      if self.morphism.map[i] == Pointed.astr: 
        refined_morphism.append(Pointed.astr)
      else: 
        alph_i = self.morphism.map[i]
        mode = refined_co_domain_rel_flat.get_mode(alph_i)
        N = refined_co_domain_rel_flat.get_prefix_length(alph_i)
        l = mode.length
        refined_morphism += [N + k for k in range(l)]
        
    return nested_tuple_morphism(refined_domain, refined_co_domain, tuple(refined_morphism))
    
  def make_pushforward(self, refined_domain:NestedTuple): 
    assert na.refines(refined_domain, self.domain) == True 
    refined_domain_rel_flat = na.get_refine_relative_flatten(refined_domain, self.domain)
    refined_co_domain_modes = [None]*self.co_domain.length
    
    for j in range(self.co_domain.length): 
      pre_j = self.morphism.preimg[j]
      if not pre_j: 
        refined_co_domain_modes[j] = self.co_domain.get_entry(j)
      else: 
        refined_co_domain_modes[j] = refined_domain_rel_flat.get_mode(pre_j[0])
      
    refined_morphism = []
    refined_co_domain = na.substitute(self.co_domain.prof, refined_co_domain_modes)
    refined_co_domain_rel_flat = na.get_refine_relative_flatten(refined_co_domain, self.co_domain)
    for i in range(refined_domain_rel_flat.rank):
      alph_i = self.morphism.map[i]
      d_mode = refined_domain_rel_flat.get_mode(i)
      if alph_i == Pointed.astr: 
        refined_morphism += [Pointed.astr]*d_mode.length 
      else: 
        cd_mode = refined_co_domain_rel_flat.get_mode(alph_i)
        alph_N = refined_co_domain_rel_flat.get_prefix_length(alph_i)
        refined_morphism += [alph_N + k for k in range(cd_mode.length)]
        
    return nested_tuple_morphism(refined_domain, refined_co_domain, tuple(refined_morphism))
        

        
      
  
  def _draw_profile(self, ax, prof, x0, y0, scale=0.6, depth=0):
    """
    Draws the Profile as vertical nested brackets.
    Returns next y position.
    """
    if prof.is_atom():
        return y0 + 1

    start = y0
    cur = y0

    for sub in prof.value:
        cur = self._draw_profile(ax, sub, x0, cur, scale, depth + 1)

    end = cur - 1

    x = x0 + depth * scale
    ax.plot([x, x], [start, end], lw=2)
    ax.plot([x, x + 0.2], [start, start], lw=2)
    ax.plot([x, x + 0.2], [end, end], lw=2)

    return cur



  def draw(self):
    import matplotlib.pyplot as plt

    S = self.domain.int_tuple
    T = self.co_domain.int_tuple
    a = self.morphism.map

    m, n = len(S), len(T)

    fig, ax = plt.subplots(figsize=(5, max(m, n) * 0.6))

    xS, xT = 1.5, 6.0

    # draw profiles
    self._draw_profile(ax, self.domain.prof, xS - 1.0, 0)
    self._draw_profile(ax, self.co_domain.prof, xT - 1.0, 0)

    # draw entries
    for i, v in enumerate(S):
        ax.text(xS, i, str(v), ha="center", va="center")

    for j, v in enumerate(T):
        ax.text(xT, j, str(v), ha="center", va="center")

    # arrows (unchanged, clean)
    for i, v in enumerate(a):
        if v != Pointed.astr:
            ax.annotate(
                "",
                xy=(xT - 0.3, v),
                xytext=(xS + 0.3, i),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_xlim(0, 7)
    ax.set_ylim(-0.5, max(m, n) - 0.5)
    ax.axis("off")
    plt.show()



  def __repr__(self):
    return (
        f"nested_tuple_morphism("
        f"domain={self.domain}, "
        f"co_domain={self.co_domain}, "
        f"map={self.morphism.map}"
        f")"
    )

def just_compose(ref_mor_1:nested_tuple_morphism, ref_mor_2:nested_tuple_morphism): 
  composed_map = []
  for i in range(ref_mor_1.domain.length): 
    alph_i = ref_mor_1.morphism.map[i]
    if alph_i == Pointed.astr: 
      composed_map.append(Pointed.astr)
    else: 
      v1_cd = ref_mor_1.co_domain.get_entry(alph_i)
      v2_d = ref_mor_2.domain.get_entry(alph_i)
      if v1_cd != v2_d: 
        return None 
      else: 
        beta_after_alph_i = ref_mor_2.morphism.map[alph_i]
        if beta_after_alph_i == Pointed.astr: 
          composed_map.append(Pointed.astr)
        else: 
          v2_cd = ref_mor_2.co_domain.get_entry(beta_after_alph_i)
          if v2_d != v2_cd: 
            return None 
          else: 
            composed_map.append(beta_after_alph_i)
            
  return nested_tuple_morphism(ref_mor_1.domain, ref_mor_2.co_domain, tuple(composed_map))

def compose_morphism(mor1:nested_tuple_morphism, mor2: nested_tuple_morphism): 
  ref_mor1_cd, ref_mor2_d = na.mutual_refinement(mor1.co_domain, mor2.domain)
  ref_mor1 = mor1.make_pullback(ref_mor1_cd)
  ref_mor2 = mor2.make_pushforward(ref_mor2_d)
  return just_compose(ref_mor1, ref_mor2)

    