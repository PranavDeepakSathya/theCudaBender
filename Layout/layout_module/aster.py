from enum import Enum, auto

class Pointed(Enum): 
  astr = auto()
  
  def __repr__ (self): 
    return ".*."