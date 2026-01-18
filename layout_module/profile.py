from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum, auto


class Atom(Enum):
  STAR = auto()

  def __repr__(self) -> str:
    return "*"


@dataclass(frozen=True)
class Profile:
  value: Atom | Tuple["Profile", ...]
  _rank: int = 0
  _length: int = 0
  _depth: int = 0

  def __post_init__(self):
    if isinstance(self.value, Atom):
      object.__setattr__(self, "_rank", 1)
      object.__setattr__(self, "_length", 1)
      object.__setattr__(self, "_depth", 0)
      return

    if isinstance(self.value, tuple):
      for p in self.value:
        if not isinstance(p, Profile):
          raise TypeError("Tuple elements must be Profile instances")
      if len(self.value) == 0:
        object.__setattr__(self, "_rank", 0)
        object.__setattr__(self, "_length", 0)
        object.__setattr__(self, "_depth", 0)
      else:
        object.__setattr__(self, "_rank", len(self.value))
        object.__setattr__(self, "_length", sum(p._length for p in self.value))
        object.__setattr__(self, "_depth", 1 + max(p._depth for p in self.value))
      return

    raise TypeError("Profile value must be Atom or tuple of Profile")

  @property
  def rank(self) -> int:
    return self._rank

  @property
  def length(self) -> int:
    return self._length

  @property
  def depth(self) -> int:
    return self._depth

  def is_atom(self) -> bool:
    return isinstance(self.value, Atom)

  def is_empty(self) -> bool:
    return self.value == ()

  def is_tuple(self) -> bool:
    return isinstance(self.value, tuple)
  
  def is_flat(self) -> bool: 
    return self.is_empty() or self.is_atom() or self._depth == 1

  def get_mode(self, i: int) -> "Profile":
    assert not self.is_empty()
    assert 0 <= i < self.rank
    if self.is_atom():
      return self
    return self.value[i]
  
  def get_length(self, i: int) -> int:
    if self.is_empty(): 
      return 0
    assert 0 <= i < self.rank
    return self.get_mode(i).length

  def get_prefix_length(self, i: int) -> int:
    assert 0 <= i <= self.rank
    if self.is_empty():
      return 0
    if self.is_atom():
      return i
    return sum(self.value[k].length for k in range(i))

  def get_suffix_length(self, i: int) -> int:
    assert 0 <= i <= self.rank
    return self.length - self.get_prefix_length(i)

  def substitute(self, profile_list: List["Profile"]) -> "Profile":
    for q in profile_list:
      assert isinstance(q, Profile)

    assert self.length == len(profile_list)

    if self.is_empty():
      return self
    if self.is_atom():
      return profile_list[0]

    curr = 0
    out = ()
    for p in self.value:
      l = p.length
      out += (p.substitute(profile_list[curr:curr + l]),)
      curr += l

    return Profile(out)

  def substitute_mode(self, i: int, qs: List["Profile"]) -> "Profile":
    assert not self.is_empty()
    assert 0 <= i < self.rank
    assert len(qs) == self.get_mode(i).length

    prefix = self.get_prefix_length(i)
    suffix = self.get_suffix_length(i + 1)

    pad = (
      [Profile(Atom.STAR)] * prefix +
      qs +
      [Profile(Atom.STAR)] * suffix
    )

    return self.substitute(pad)

  def __eq__(self, other) -> bool:
    return isinstance(other, Profile) and self.value == other.value

  def __repr__(self) -> str:
    if self.is_atom():
      return repr(self.value)
    return "(" + ", ".join(repr(p) for p in self.value) + ")"
