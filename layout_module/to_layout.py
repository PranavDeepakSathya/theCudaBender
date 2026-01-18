from.layout import Layout
from .composition import nested_tuple_morphism
from .aster import Pointed
from functools import reduce
from operator import mul
from .nested_tuple import NestedTuple
from . import flat_algebra as fa
def to_layout(X,canonicize = True):
  s,t,prof = X.domain.int_tuple, X.co_domain.int_tuple, X.domain.prof
  d = []
  for i in range(len(s)):
    v = X.morphism.map[i] 
    if v == Pointed.astr: 
      d.append(0)
    else: 
      d.append(reduce(mul,t[0:v],1))
  
    
  shape = NestedTuple(s,prof)
  stride = NestedTuple(d, prof)
  
  L = Layout(shape, stride)
  if canonicize:
    return (L.squeeze()).filter_zeros()