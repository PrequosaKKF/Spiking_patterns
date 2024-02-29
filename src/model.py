import numpy as np
import neuron as nr
from neuron.units import ms, mV
#units: [v] = [v_inf] = mV, [tau] = ms, [g] = uS/mm2, [A] = mm2, [i_ext] = nA, [c] = nF/mm2

class HodgkinHuxleyModel():
  def __init__(self):
    self.params  = [[self.a_m1, self.a_m2, self.a_m3, self.a_m4, self.a_m5, self.b_m1, self.b_m2, self.b_m3],
        [self.a_n1, self.a_n2, self.a_n3, self.a_n4, self.a_n5, self.b_n1, self.b_n2, self.b_n3],
        [self.a_h1, self.a_h2, self.a_h3, self.b_h1, self.b_h2, self.b_h3, self.b_h4, self.amp]] \
        = [[0.1, 40, 1, -0.1, 40, 4, -0.0556, 65],
        [0.01, 55, 1, -0.1, 55, 0.125, -0.0125, 65],
        [0.07, -0.05, 65, 1, 1, -0.1, 35, 200]]
    self.vs = [self.v_l, self.v_k, self.v_na] = np.array([-54.387, -77, 50])
    self.gs_max = [self.g_l, self.g_k, self.g_na] = np.array([3, 360, 1200])
    self.m,self.n,self.h = 5.29550886e-02,  3.17732346e-01,  5.95992386e-01
    self.ps = [p_l, p_k, p_na] = np.array([1, self.n**4, self.m**3*self.h])
    self.gs = self.gs_max*self.ps
    self.area = 1
    self.cm = 10
    self.dur = 250
    self.delay = 10
    self.dt = .025
    self.v = -65
    self.t = 0
    return None
  def i_ext(self, t):
    return self.amp if t>=self.delay and t<=(self.delay+self.dur) else 0
  def a_m(self, v):
    return self.a_m1*(v+self.a_m2) / (self.a_m3-np.exp(self.a_m4*(v+self.a_m5)))
  def a_n(self, v):
    return self.a_n1*(v+self.a_n2) / (self.a_n3-np.exp(self.a_n4*(v+self.a_n5)))
  def a_h(self, v):
    return self.a_h1 * np.exp(self.a_h2*(v+self.a_h3))
  def b_m(self, v):
    return self.b_m1 * np.exp(self.b_m2*(v+self.b_m3))
  def b_n(self, v):
    return self.b_n1 * np.exp(self.b_n2*(v+self.b_n3))
  def b_h(self, v):
    return self.b_h1 / (self.b_h2+np.exp(self.b_h3*(v+self.b_h4)))
  def tau_m(self, v):
    return 1 / (self.a_m(v)+self.b_m(v))
  def tau_n(self, v):
    return 1 / (self.a_n(v)+self.b_n(v))
  def tau_h(self, v):
    return 1 / (self.a_h(v)+self.b_h(v))
  def m_inf(self, v):
    return self.a_m(v) * self.tau_m(v)
  def n_inf(self, v):
    return self.a_n(v) * self.tau_n(v)
  def h_inf(self, v):
    return self.a_h(v) * self.tau_h(v)
  
  def Initialize(self, v):
    self.m = self.m_inf(self.v) + (self.m-self.m_inf(self.v))*np.exp(-self.dt/2/self.tau_m(self.v))
    self.n = self.n_inf(self.v) + (self.n-self.n_inf(self.v))*np.exp(-self.dt/2/self.tau_n(self.v))
    self.h = self.h_inf(self.v) + (self.h-self.h_inf(self.v))*np.exp(-self.dt/2/self.tau_h(self.v))
    steady = False
    while not steady:
      tmp_v = self.v
      self.ps = np.array([1, self.n**4, self.m**3*self.h])
      self.gs = self.gs_max*self.ps
      self.g = np.sum(self.gs)
      self.tau_v = self.cm*self.area / self.g
      self.v_inf = (np.sum(self.gs*self.vs) + self.i_ext(self.t)/self.area) / self.g
      self.i = np.sum(self.gs*(self.v-self.vs))
      self.v = self.v_inf + (self.v-self.v_inf)*np.exp(-self.dt/self.tau_v)
      
      self.m = self.m_inf(self.v) + (self.m-self.m_inf(self.v))*np.exp(-self.dt/self.tau_m(self.v))
      self.n = self.n_inf(self.v) + (self.n-self.n_inf(self.v))*np.exp(-self.dt/self.tau_n(self.v))
      self.h = self.h_inf(self.v) + (self.h-self.h_inf(self.v))*np.exp(-self.dt/self.tau_h(self.v))

      self.y = np.array([self.v,self.m,self.n,self.h]).reshape((1,4))
      self.y_inf = np.array([self.v_inf,self.m_inf(self.v),self.n_inf(self.v),self.h_inf(self.v)]).reshape((1,4))
      if abs(self.v-tmp_v) <= 1e-6: steady = True
    return None
  
  def ContinueRun(self, t_end):
    self.js = np.array([0])
    self.ys = np.array([self.v,self.m,self.n,self.h]).reshape((1,4))
    self.ts = np.array([self.t])
    self.ys_inf = np.array([self.v,self.m,self.n,self.h]).reshape([1,4])

    while self.t < t_end:
      self.ps = np.array([1, self.n**4, self.m**3*self.h])
      self.gs = self.gs_max*self.ps
      self.g = np.sum(self.gs)
      self.tau_v = self.cm*self.area / self.g
      self.v_inf = (np.sum(self.gs*self.vs) + self.i_ext(self.t)/self.area) / self.g
      self.i = np.sum(self.gs*(self.v-self.vs))
      self.v = self.v_inf + (self.v-self.v_inf)*np.exp(-self.dt/self.tau_v)
      
      self.m = self.m_inf(self.v) + (self.m-self.m_inf(self.v))*np.exp(-self.dt/self.tau_m(self.v))
      self.n = self.n_inf(self.v) + (self.n-self.n_inf(self.v))*np.exp(-self.dt/self.tau_n(self.v))
      self.h = self.h_inf(self.v) + (self.h-self.h_inf(self.v))*np.exp(-self.dt/self.tau_h(self.v))

      self.t += self.dt

      self.js = np.append(self.js, [self.i])
      self.y = np.array([self.v,self.m,self.n,self.h]).reshape((1,4))
      self.y_inf = np.array([self.v_inf,self.m_inf(self.v),self.n_inf(self.v),self.h_inf(self.v)]).reshape((1,4))
      self.ys = np.append(self.ys, self.y, axis=0)
      self.ys_inf = np.append(self.ys_inf, self.y_inf, axis=0)
      self.ts = np.append(self.ts, self.t)
    return None
  
class IzhikevichModel():
    def __init__(self, params):
        self.k = params["k"]
        self.a = params["a"]
        self.b = params["b"]
        self.d = params["d"]
        self.C = params["C"]
        self.Vr = params["Vr"]
        self.Vt = params["Vt"]
        self.Vp = params["Vp"]
        self.c = params["c"]
        self.dt = .025
        self.v = self.Vr
        self.u = 0
        self.t = 0
        self.vs = np.array([self.v])
        self.ts = np.array([self.t])

        self.amp = params['I']
        self.delay = 2
        self.dur = 40
        return None
    def I(self, t):
        return self.amp if (t > self.delay) and (t < self.dur + self.delay) else 0
    def Dv(self, v, u, i):
        return (1/self.C)*(self.k*(v-self.Vr)*(v-self.Vt) - u + i)
    def Du(self, v, u):
        return self.a*(self.b*(v-self.Vr) - u)
    def Initialize(self, v):
        self.v = v
        self.u = 0
        self.t = 0
        self.vs = np.array([self.v])
        self.ts = np.array([self.t])
        steady = False
        while (not steady):
            tmp_v = self.v
            tmp_u = self.u
            if (self.v >= self.Vp) :
                self.v = self.c
                self.u += self.d
            else:
                self.v += self.Dv(self.v, self.u, self.I(self.t))*self.dt
                self.u += self.Du(self.v, self.u)*self.dt
            if (np.abs(tmp_v - self.v) < 1e-5) and (np.abs(tmp_u - self.u) < 0.01):
                steady = True
        return None
    def ContinueRun(self, t_end):
        while (self.t < t_end):
            if (self.v >= self.Vp) :
                self.v = self.c
                self.u += self.d
            else:
                self.v += self.Dv(self.v, self.u, self.I(self.t))*self.dt
                self.u += self.Du(self.v, self.u)*self.dt
            self.t += self.dt
            self.vs = np.append(self.vs, self.v)
            self.ts = np.append(self.ts, self.t)
        return None