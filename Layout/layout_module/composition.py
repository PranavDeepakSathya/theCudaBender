from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul
from collections import defaultdict
from .flat_algebra import Pointed

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


    