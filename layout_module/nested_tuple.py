from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum, auto
from functools import cached_property
from functools import reduce
from operator import mul

from .profile import Profile, Atom

@dataclass(frozen=True)
class NestedTuple:
  int_tuple: tuple[int]
  prof: Profile
  _rank: int = 0
  _length: int = 0
  _depth: int = 0
  _size: int = 0
  
  @staticmethod
  def from_literal(obj) -> "NestedTuple":
    def walk(x):
      if isinstance(x, int):
        return Profile(Atom.STAR), [x]

      if isinstance(x, tuple):
        if len(x) == 0:
          return Profile(()), []

        parts = []
        flat = []
        for y in x:
          p, f = walk(y)
          parts.append(p)
          flat.extend(f)
        return Profile(tuple(parts)), flat

      raise TypeError(f"Invalid literal for NestedTuple: {x!r}")

    prof, flat = walk(obj)
    return NestedTuple(tuple(flat), prof)

  def __post_init__(self):
    assert isinstance(self.prof, Profile)
    assert self.prof.length == len(self.int_tuple)

    object.__setattr__(self, "_rank", self.prof.rank)
    object.__setattr__(self, "_length", self.prof.length)
    object.__setattr__(self, "_depth", self.prof.depth)

    if len(self.int_tuple) == 0:
      object.__setattr__(self, "_size", 0)
    else:
      object.__setattr__(self, "_size", reduce(mul, self.int_tuple, 1))

  @property
  def rank(self) -> int:
    return self._rank

  @property
  def length(self) -> int:
    return self._length

  @property
  def depth(self) -> int:
    return self._depth

  @property
  def size(self) -> int:
    return self._size
  
  def is_flat(self) -> bool: 
    return self.prof.is_flat()
  
  def flatten(self) -> bool: 
     new_prof = Profile(tuple([Profile(Atom.STAR)]*self.length))
     return NestedTuple(self.int_tuple, new_prof)
    

  def get_prefix_length(self, i: int) -> int:
    return self.prof.get_prefix_length(i)

  def get_suffix_length(self, i: int) -> int:
    return self.prof.get_suffix_length(i)

  def get_prefix_size(self, i: int) -> int:
    assert 0 <= i <= self.rank
    if i == 0:
      return 0
    return reduce(mul, self.int_tuple[:self.get_prefix_length(i)], 1)

  def get_suffix_size(self, i: int) -> int:
    assert 0 <= i <= self.rank
    if i == self.rank:
      return 0
    return reduce(mul, self.int_tuple[self.get_prefix_length(i):], 1)
  
  def get_length(self, i: int) -> int: 
    assert 0 <= i < self.rank 
    return self.prof.get_length(i)
  
  def get_mode(self, i: int) -> "NestedTuple":
    assert 0 <= i < self.rank
    start = self.get_prefix_length(i)
    length = self.get_length(i)
    int_tuple_mode = self.int_tuple[start:start + length]
    prof_mode = self.prof.get_mode(i)
    return NestedTuple(int_tuple_mode, prof_mode)
  
  def get_entry(self, i:int) -> int: 
    assert 0 <= i < self.length
    return NestedTuple((self.int_tuple[i],), Profile(Atom.STAR))

  def _build_from_flat(self, p: Profile, data: tuple, i: int = 0):
    if p.is_empty():
      return (), i
    if p.is_atom():
      return data[i], i + 1

    out = []
    for q in p.value:
      v, i = self._build_from_flat(q, data, i)
      out.append(v)
    return tuple(out), i

  def __repr__(self) -> str:
    v, _ = self._build_from_flat(self.prof, self.int_tuple)
    return repr(v)

  def pretty_print(self, indent: int = 0, is_last: bool = True):
    pad = "  " * indent
    branch = "└─ " if is_last else "├─ "

    if indent == 0:
      print(
        f"NestedTuple "
        f"(rank={self.rank}, length={self.length}, "
        f"depth={self.depth}, size={self.size})"
      )
      print(f"value = {self}")
    else:
      print(
        f"{pad}{branch}NestedTuple "
        f"(rank={self.rank}, length={self.length}, "
        f"depth={self.depth}, size={self.size})"
      )
      print(f"{pad}{'   ' if is_last else '│  '}value = {self}")

    if self.prof.is_atom() or self.prof.is_empty():
      return

    for i in range(self.rank):
      last = (i == self.rank - 1)
      edge_pad = pad + ("   " if is_last else "│  ")

      print(
        f"{edge_pad}{'└─' if last else '├─'} "
        f"mode {i} "
        f"[prefix_len={self.get_prefix_length(i)}, "
        f"prefix_size={self.get_prefix_size(i)}]"
      )

      self.get_mode(i).pretty_print(
        indent + 2,
        is_last=last
      )

  def __eq__ (self, other): 
    return self.int_tuple == other.int_tuple and self.prof == other.prof 
  
  