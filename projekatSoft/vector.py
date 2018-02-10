# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 02:15:35 2018

@author: Saska
"""


import math

# http://www.fundza.com/vectors/point2line/index.html

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
  
def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)
  
def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
  
def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)
  
def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

  
def pnt2line(pnt, start, end):
    s = start
    e = end
    p=pnt
    line_vec = vector(s, e)
    lv = line_vec
    pnt_vec = vector(s, p)
    line_len = length(lv)
    line_unitvec = unit(lv)
    a=1.0/line_len
    pnt_vec_scaled = scale(pnt_vec, a)
    t = dot(line_unitvec, pnt_vec_scaled)    
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(lv, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)




