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
        refined_morphism += [N + i for i in range(l)]
        
    return nested_tuple_morphism(refined_domain, refined_co_domain, tuple(refined_morphism))
    
      
  
  def _draw_bracket(self, ax, x, y0, y1, width=0.15, lw=1.2):
    # vertical line
    ax.plot([x, x], [y0, y1], linewidth=lw)
    # top hook
    ax.plot([x, x + width], [y1, y1], linewidth=lw)
    # bottom hook
    ax.plot([x, x + width], [y0, y0], linewidth=lw)

  def _draw_nested_brackets(self, ax, nt: NestedTuple, x, y_offset=0):
    """
    Draw brackets for a NestedTuple using its modes.
    Returns total flattened height (= nt.length).
    """
    # base case: atomic profile
    if nt.prof.is_atom():
        return 1

    start = y_offset
    current = y_offset

    for i in range(nt.rank):
        mode = nt.get_mode(i)
        h = self._draw_nested_brackets(ax, mode, x, current)
        current += h

    end = current - 1
    self._draw_bracket(ax, x, start, end)
    return current - y_offset

  
  def draw(self):
    

    S = self.domain.int_tuple
    T = self.co_domain.int_tuple
    a = self.morphism.map

    m = len(S)
    n = len(T)

    fig, ax = plt.subplots(figsize=(4, max(m, n) * 0.6))

    xS = 0
    xT = 3
    xSb = -0.6
    xTb = 2.4

    yS = list(range(m))
    yT = list(range(n))

    for i, y in enumerate(yS):
        ax.text(xS, y, str(S[i]), ha="center", va="center")

    for j, y in enumerate(yT):
        ax.text(xT, y, str(T[j]), ha="center", va="center")

    pad = 0.25
    for i, v in enumerate(a):
        if v != Pointed.astr:
            ax.annotate(
                "",
                xy=(xT - pad, yT[v]),
                xytext=(xS + pad, yS[i]),
                arrowprops=dict(arrowstyle="->", linewidth=1.5),
            )

    # structure
    self._draw_nested_brackets(ax, self.domain, xSb)
    self._draw_nested_brackets(ax, self.co_domain, xTb)

    ax.set_xlim(-1.2, 3.2)
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
