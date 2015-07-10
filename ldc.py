#! /usr/bin/python
# -*- coding: utf-8 -*-
# File: ldc.py
"""2d lid driven cavity for compressible flow

Creates a "plot.jpeg" and a "ghia.jpeg" containing the most 
interesting visualization. Example files should be included. 

This code was written as a part of a CFD course on turbulence and DNS 
given at AALTO university, Finland, in 2015. Being a course exercise 
this code was never rigorously tested but according to the test data by
Ghia et al. (1982) (function ghia in this code) it works fine. I'm just 
publicing this out of a habit. Someone might find it useful while learning 
Python. A short given on the subject containing some of the equations should be 
included.

Lisence: DO WHATEVER YOU WANT. A Citation would be appreciated.

Written by Antti Mikkonen, a.mikkonen@iki.fi, 2015

"""
from __future__ import division

__author__="Antti Mikkonen, a.mikkonen@iki.fi"
__date__ ="$2015$"

# ****************************************************************************
import pprint
pp = pprint.PrettyPrinter()
def pprint(stuff):
    pp.pprint(stuff)

import pickle
from time import time
from copy import copy
import numpy as np
import scipy.sparse as sparse
import matplotlib
from matplotlib import pylab as plt
matplotlib.rc('font',
                        **{'family'     :   'sans-serif',
                           'sans-serif' :   ['Helvetica'],
                           'size'       :   12
                           }
                      )
matplotlib.rc("lines", 
                **{'linewidth'          :   0.5,
                   'markersize'         :   2,
                   'markeredgewidth'    :   0.5,
                   }
              )
matplotlib.rc("axes",
                **{'linewidth' : 0.5}
              )
matplotlib.rc("text", 
                **{"usetex" : True}
              )
inch = 2.54
x_normal, y_normal = 7.8/inch,5.73/inch
x_full = 16/inch        
# ****************************************************************************
# ****************************************************************************
class LDC(object):
    def __init__(self):
        self.start = time()
        # Given parameters
        self.L              = 1
        self.nx             = 64#128
        self.Re             = 400 
        self.u_lid          = 0.3*300
        self.nu             = self.u_lid * self.L / self.Re 

        self.dt             = 1e-6 
        self.time_end       = 1000
        self.write_interval = 1000 
        self.save_interval  = 10000
        self.init           = False #"last128" #False #

        self.ramp   = 1
        self.cutter = 0.07 

        # Initialization
        self.rho0  = 1.205
        self.mu    = self.nu * self.rho0   
        self.p0    = 1e5
        self.u0    = 0
        self.v0    = 0
        self.gamma = 1.4
         
        # Calculated parameters
        self.calculate_parameters()

        # Initialize fields
        self.initialize(self.init)
        
    def solve(self):
        self._build_operators()
        
        self.post()
        self.plot()

        while self.time < self.time_end:
#             self._rk4()
            self._euler()
            if self.time_step % 100 == 0:
                self._filter()
              
            if self.time_step % self.write_interval== 0:
                print self.time_step, self.time
            self.time_step += 1
            self.time = self.time_step * self.dt
            
            if self.time_step % self.save_interval == 0:
                self.post()
                self.plot()
                self.save_fields(self.time)
        
        # End
        self.save_fields("last")
        self.post()
        self.plot()
        
    def _filter(self):
        cut = self.cutter * (self.h/np.pi)**2
        self.rho  += cut * self.laplacianFilter * self.rho
#         self.rhoU += cut * self.laplacianFilter * self.rhoU
#         self.rhoV += cut * self.laplacianFilter * self.rhoV
#         self.u    += cut * self.laplacianFilter * self.u
#         self.v    += cut * self.laplacianFilter * self.v
        self.p    += cut * self.laplacianFilter * self.p
        self.e    += cut * self.laplacianFilter * self.e

    def post(self):
        self.U   = self.u.reshape((self.nx,self.nx))
        self.V   = self.v.reshape((self.nx,self.nx))
        self.P   = self.p.reshape((self.nx,self.nx))
        self.RHO = self.rho.reshape((self.nx,self.nx))
        
        print
        print "Statistics"
        print "end time", self.time
        print "steps   ", self.time_step
        print "dt      ", self.dt
        print "h       ", self.h
        print
        print "Co      ", self.Co
        print "clf diff", self.clf_diff
        print "clf soni", self.clf_sonic
        print
        print "u lid   ", self.u_lid
        print "Re      ", self.Re
        print "nu      ", self.nu
        print "mu      ", self.mu
        print
        print "max u   ", self.u.max() / self.u_lid
        print "min u   ", self.u.min() / self.u_lid
        print "max v   ", self.v.max() / self.u_lid
        print "min v   ", self.v.min() / self.u_lid
        print "max rho ", self.rho.max() 
        print "min rho ", self.rho.min()
        print "max p   ", self.p.max() 
        print "min p   ", self.p.min() 
        print 
    
    def save_fields(self, end):
        ofile = open("fields_" + str(end), "w+")
        fields = (self.rho, self.rhoU, self.rhoV, 
                  self.e, self.p, self.u, self.v)
        pickle.dump(fields, ofile)
        ofile.close()
    
    def load_fields(self, end):
        path = "fields_" + str(end)
        print "Loading ", path
        ifile = open(path, "r+")
        self.rho, self.rhoU, self.rhoV, self.e,\
                             self.p, self.u, self.v = pickle.load(ifile) 
        ifile.close()
    
    def initialize(self, end=False):
        self.rho  = np.ones(self.nx**2,dtype='float64') * self.rho0
        self.p    = np.ones(self.nx**2,dtype='float64') * self.p0
        self.u    = np.ones(self.nx**2,dtype='float64') * self.u0
        self.v    = np.ones(self.nx**2,dtype='float64') * self.v0
        
        self.rhoU = self.rho * self.u
        self.rhoV = self.rho * self.v
        self.e    = self.p / (self.gamma - 1) + 0.5 * self.rho \
                                                * (self.u**2 + self.v**2)
        if end:
            self.load_fields(end)
        
    def calculate_parameters(self):
        self.h = self.L / self.nx
        self.Co     = self.u_lid * self.dt / self.h
        self.clf_diff    = self.dt*self.nu/self.h**2
        self.clf_sonic    = 300 * self.dt / self.h
        
        assert self.clf_diff  < 0.25
        assert self.Co        < 0.25
        assert self.clf_sonic < 0.25
        
        # Arrays
        self.x      = np.linspace(self.h/2, self.L-self.h/2, 
                                   self.nx,dtype='float64')
        self.y      = np.linspace(self.h/2, self.L-self.h/2, 
                                   self.nx,dtype='float64')
        # For convenience
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Indexes
        self.Left_index  = np.arange(0,self.nx**2, self.nx)
        self.Right_index = np.arange(self.nx-1, self.nx**2, self.nx)
        self.Down_index  = np.arange(self.nx)
        self.Up_index    = np.arange(self.nx**2-self.nx,self.nx**2)
        
        # Counters
        self.time_step = 0
        self.time      = 0
    
    def _build_dxdy(self):
        """
        Wasteful implementation as the same indexes are repeated. 
        Is run only once, performance is not important. 
        """
        
        xvals = []
        xrows = []
        xcols = []
        yvals = []
        yrows = []
        ycols = []
        
        C = (2 * self.h)**-1
        for row in range(self.nx):
            for col in range(self.nx):
                matrix_index = row*self.nx + col
                
                # left
                if col > 0:
                    # middle
                    xrows.append(matrix_index)
                    xcols.append(matrix_index)
                    xvals.append(-C)
                    # left
                    xrows.append(matrix_index)
                    xcols.append(matrix_index-1)
                    xvals.append(-C)
                # right
                if col < self.nx -1 :
                    # middle
                    xrows.append(matrix_index)
                    xcols.append(matrix_index)
                    xvals.append(C)
                    # right
                    xrows.append(matrix_index)
                    xcols.append(matrix_index+1)
                    xvals.append(C)
                 
                # down
                if row > 0:
                    # middle
                    yrows.append(matrix_index)
                    ycols.append(matrix_index)
                    yvals.append(-C)
                    # down
                    yrows.append(matrix_index)
                    ycols.append(matrix_index-self.nx)
                    yvals.append(-C)
                # up
                if row < self.nx - 1:
                    # middle
                    yrows.append(matrix_index)
                    ycols.append(matrix_index)
                    yvals.append(C)
                    # down
                    yrows.append(matrix_index)
                    ycols.append(matrix_index+self.nx)
                    yvals.append(C)
        
        # No boundary effects       
        self.dx = sparse.csr_matrix(sparse.coo_matrix((xvals, (xrows, xcols))))
        self.dy = sparse.csr_matrix(sparse.coo_matrix((yvals, (yrows, ycols))))        

        ## Zero-gradient boundaries
        # Right boundary
        for k in self.Right_index:
            xrows.append(k)
            xcols.append(k)
            xvals.append(2*C)
        # Left boundary
        for k in self.Left_index:
            xrows.append(k)
            xcols.append(k)
            xvals.append(-2*C)
        self.dxZeroGrad = sparse.csr_matrix(
                                    sparse.coo_matrix((xvals, (xrows, xcols))))
        
        # Up boundary                
        for k in self.Up_index:
            yrows.append(k)
            ycols.append(k)
            yvals.append(2*C)
        # Down boundary
        for k in self.Down_index:
            yrows.append(k)
            ycols.append(k)
            yvals.append(-2*C)
        self.dyZeroGrad = sparse.csr_matrix(
                                    sparse.coo_matrix((yvals, (yrows, ycols))))

    def _build_laplacian(self):
        
        # Same for both
        vals = []
        rows = []
        cols = []
        
        C = self.h**-2
        for row in range(self.nx):
            for col in range(self.nx):
                matrix_index = row*self.nx + col
                
                # left
                if col > 0:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-C)
                    # left
                    rows.append(matrix_index)
                    cols.append(matrix_index-1)
                    vals.append(C)
                    
                # right
                if col < self.nx -1 :
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-C)
                    # right
                    rows.append(matrix_index)
                    cols.append(matrix_index+1)
                    vals.append(C)

                # down
                if row > 0:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-C)
                    # down
                    rows.append(matrix_index)
                    cols.append(matrix_index-self.nx)
                    vals.append(C)
                # up
                if row < self.nx - 1:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-C)
                    # up
                    rows.append(matrix_index)
                    cols.append(matrix_index+self.nx)
                    vals.append(C)

        self.laplacianFilter = sparse.csr_matrix(
                            sparse.coo_matrix((vals, (rows, cols))))

        ## Boudaries
        for row in range(self.nx):
            for col in range(self.nx):
                matrix_index = row*self.nx + col                    
                
                # left
                if col == 0:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-2*C)
                    
                # right
                if col == self.nx -1 :
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-2*C)

                # down
                if row == 0:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-2*C)
                # up
                if row == self.nx - 1:
                    # middle
                    rows.append(matrix_index)
                    cols.append(matrix_index)
                    vals.append(-2*C)
                
        self.laplacian = sparse.csr_matrix(
                            sparse.coo_matrix((vals, (rows, cols))))
                 
    def _build_sources(self):
        """Source term for lid u velocity
        """
        self.bRhoU = np.zeros(self.nx**2,dtype='float64')
        self.bRhoU[self.Up_index] += self.mu * self.h**-2 * 2 \
                                        * self.ramp * self.u_lid                               

    def _build_operators(self):
        self._build_dxdy()
        self._build_laplacian()
        self._build_sources()
        
    def _dNS(self, rho, rhoU, rhoV, e, p, u, v):
        ## TODO: awfully wasteful implementation  
        
        drho  = self.dt * ( - self.dx * rhoU 
                            - self.dy * rhoV
                            ) 

        drhoU = self.dt * ( - self.dx * (rhoU * u) 
                            - self.dy * (rhoU * v) 
                            - self.dxZeroGrad * p 
                            + self.mu * self.laplacian * u 
                            + self.bRhoU # Source term
                            )
        
        drhoV = self.dt * ( - self.dx * (rhoV * u) 
                            - self.dy * (rhoV * v) 
                            - self.dyZeroGrad * p
                            + self.mu * self.laplacian * v
                            )

        de    = self.dt * ( - self.dx * ( u * (e + p))
                            - self.dy * ( v * (e + p)) 
                           )   
        ## Update 
        rho        += drho
        rhoU       += drhoU
        rhoV       += drhoV
        e          += de
        ## Update primitives
        du          = rhoU / rho - u
        dv          = rhoV / rho - v    
        u           = u + du
        v           = v + dv
        dp = (e - 0.5 * self.rho * (self.u**2 + self.v**2)) \
                * (self.gamma - 1) - p
        
        return drho, drhoU, drhoV, de, dp, du, dv

    def _euler(self):
        ## TODO: awfully wasteful implementation  
        drho, drhoU, drhoV, de, dp, du, dv = \
                                     self._dNS(   
                                                 copy(self.rho), 
                                                 copy(self.rhoU), 
                                                 copy(self.rhoV), 
                                                 copy(self.e), 
                                                 copy(self.p), 
                                                 copy(self.u), 
                                                 copy(self.v)
                                             )
        # New time
        self.rho       += drho 
        self.rhoU      += drhoU
        self.rhoV      += drhoV
        self.e         += de
        self.p         += dp
        self.u         += du
        self.v         += dv   

    def _rk4(self):
        ## TODO: awfully wasteful implementation  
        
        # k1
        drho1, drhoU1, drhoV1, de1, dp1, du1, dv1 = \
                                     self._dNS(   
                                                 copy(self.rho), 
                                                 copy(self.rhoU), 
                                                 copy(self.rhoV), 
                                                 copy(self.e), 
                                                 copy(self.p), 
                                                 copy(self.u), 
                                                 copy(self.v)
                                             )
        #k2
        drho2, drhoU2, drhoV2, de2, dp2, du2, dv2 = \
                                    self._dNS( 
                                      self.rho  + 0.5  * drho1, 
                                      self.rhoU + 0.5  * drhoU1,
                                      self.rhoV + 0.5  * drhoV1,
                                      self.e    + 0.5  * de1,
                                      self.p    + 0.5  * dp1,
                                      self.u    + 0.5  * du1,
                                      self.v    + 0.5  * dv1
                                      ) 
        #k3
        drho3, drhoU3, drhoV3, de3, dp3, du3, dv3 = \
                                    self._dNS( 
                                      self.rho  + 0.5  * drho2, 
                                      self.rhoU + 0.5  * drhoU2,
                                      self.rhoV + 0.5  * drhoV2,
                                      self.e    + 0.5  * de2,
                                      self.p    + 0.5  * dp2,
                                      self.u    + 0.5  * du2,
                                      self.v    + 0.5  * dv2
                                      ) 
        #k4
        drho4, drhoU4, drhoV4, de4, dp4, du4, dv4 = \
                                    self._dNS( 
                                      self.rho  + 0.5  * drho3, 
                                      self.rhoU + 0.5  * drhoU3,
                                      self.rhoV + 0.5  * drhoV3,
                                      self.e    + 0.5  * de3,
                                      self.p    + 0.5  * dp3,
                                      self.u    + 0.5  * du3,
                                      self.v    + 0.5  * dv3
                                      )
        # New time
        self.rho       += 1 / 6 * (drho1  + 2*drho2  + 2*drho3  + drho4) 
        self.rhoU      += 1 / 6 * (drhoU1 + 2*drhoU2 + 2*drhoU3 + drhoU4)
        self.rhoV      += 1 / 6 * (drhoV1 + 2*drhoV2 + 2*drhoV3 + drhoV4)
        self.e         += 1 / 6 * (de1    + 2*de2    + 2*de3    + de4)
        self.p         += 1 / 6 * (dp1    + 2*dp2    + 2*dp3    + dp4)
        self.u         += 1 / 6 * (du1    + 2*du2    + 2*du3    + du4)
        self.v         += 1 / 6 * (dv1    + 2*dv2    + 2*dv3    + dv4)   
        
    def ghia(self, Re = 400):
        """Comparison data on vertical and horizontal center line
        
        Ghia et al. (1982)
        
        """
        y = np.array([0,0.0547,0.0625,0.0703,0.1016,0.1719,
                      0.2812,0.4531,0.5000,0.6172,0.7344,
                      0.8516,0.9531,0.9609,0.9688,0.9766 
                      ])  
        if Re == 100:
            u = np.array([
                          0,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,
                          -0.1566,-0.2109,-0.2058,-0.1364,0.0033,
                          0.2315 ,0.6872,0.7372,0.7887,0.8412
                          ])
        elif Re==400:
            u = np.array([
                          0,-0.0819,-0.0927,-0.1034,-0.1461,-0.2430,
                          -0.3273,-0.1712,-0.1148,0.0214,0.1626,0.2909, 
                          0.5589,0.6176,0.6844,0.7582
                          ])
        else:
            raise ValueError("Unkonwn Re")
        
        x = np.array([
                      0,0.0625,0.0703,0.0781,0.0938,0.1563,0.2266,0.2344,     
                      0.5,0.8047,0.8594,0.9063,0.9453,0.9531,0.9609,0.9688,
                      1.0000  
                      ])
        
        if Re == 100:
            v = np.array([
                          0,0.0923,0.1009,0.1089,0.1232,0.1608,0.1751,0.1753,      
                          0.0545,-0.2453,-0.2245,-0.1691,-0.1031,-0.0886,     
                          -0.0739,-0.0591,0.0000   
                          ])
        elif Re==400:
            v = np.array([0.0000,0.1836,0.1971,0.2092,0.2297,0.2812,0.3020,   
                          0.3017,0.0519,-0.3860,-0.4499,-0.3383,-0.2285,   
                          -0.1925,-0.1566,-0.1215,0.0000
                          ])
        else:
            raise ValueError("Unkonwn Re")
        
        return x,y, u, v
    
    def plot(self):
        ####################################
        ## imshows of u, v, p, rho
        ####################################
        
        # Ignores the boundaries
        plt.clf()
        fig, ax = plt.subplots(2,2,sharey=True)

        # u 
        im = ax[0,0].imshow(self.U, 
                       extent=[0, self.L, 0, self.L],
                       origin='lower')
        im.set_interpolation('bilinear')
        ax[0,0].set_title(r"u (m/s) ")
        # v
        im = ax[0,1].imshow(self.V, 
                       extent=[0, self.L, 0, self.L],
                       origin='lower')
        im.set_interpolation('bilinear')
        ax[0,1].set_title(r"v (m/s) ")
        # p
        im = ax[1,0].imshow(self.P * 1e-5, 
                       extent=[0, self.L, 0, self.L],
                       origin='lower')
        im.set_interpolation('bilinear')
        ax[1,0].set_title(r"p  (bar)")
        # rho
        im = ax[1,1].imshow(self.RHO, 
                       extent=[0, self.L, 0, self.L],
                       origin='lower')
        im.set_interpolation('bilinear')
        ax[1,1].set_title("rho (kg/m^3) ")
         
        # Save
        fig.set_size_inches(x_full, x_full)
        fig.subplots_adjust(wspace = 0.3, hspace = 0.3)
        fig.savefig("plot.jpeg", dpi=600,bbox_inches='tight')
        
        ####################################
        ## QUIVER
        ####################################
        fig, ax = plt.subplots()
        ax.quiver( self.U, self.V)
        fig.set_size_inches(x_normal, x_normal)
        fig.savefig("quiver.jpeg", dpi=600,bbox_inches='tight')

        ####################################
        ## GHIA TEST DATA
        ####################################
        u_cfd = (self.U[:,int(np.floor(self.nx/2))] \
                    + self.U[:,int(np.ceil(self.nx/2))]) / 2 / self.u_lid
                    
        v_cfd = (self.V[int(np.floor(self.nx/2)),:] \
                    + self.V[int(np.ceil(self.nx/2)),:]) / 2 / self.u_lid
        

        x_g,y_g, u_g, v_g = self.ghia(self.Re)
        fig, ax = plt.subplots(1)
        
        ax.plot([0,self.L], [0,0], "k--")
        ax.plot(y_g, u_g, "bx",       label="ghia    u Re=" + str(self.Re))
        ax.plot(self.y, u_cfd, "b.-", label="present u Re=" + str(self.Re))
        ax.set_xlabel(r"$x/y$")
        ax.set_ylabel(r"$u^*/v^*$")
        
        ax.plot(x_g, v_g, "rx",       label="ghia    v Re=" + str(self.Re))
        ax.plot(self.x, v_cfd, "r.-", label="present v Re=" + str(self.Re))
        ax.legend(prop={'size':8},loc="upper left")     
                
        # Save
        fig.set_size_inches(x_full/2, x_full/2)
        fig.subplots_adjust(wspace = 0.3, hspace = 0.3)
        fig.savefig("ghia.jpeg", dpi=600,bbox_inches='tight')
        
        
        ####################################
        ## PRESSURE AND DENSTIY
        ####################################
        p_cfd = (self.P[:,int(np.floor(self.nx/2))] \
                    + self.P[:,int(np.ceil(self.nx/2))]) / 2 - self.p0 
                    
        rho_cfd = (self.RHO[:,int(np.floor(self.nx/2))] \
                    + self.RHO[:,int(np.ceil(self.nx/2))]) / 2 - self.rho0

        fig, ax = plt.subplots(1,2)
        
        ax[0].plot([0,self.L], [0,0], "k--")
        ax[0].plot(self.y, p_cfd*1e-5 , "b.-", label="$p$")
        ax[0].set_xlabel(r"$y$")
        ax[0].set_ylabel(r"$p-p_0 (bar)$")

        ax[1].plot([0,self.L], [0,0], "k--")
        ax[1].plot(self.y, rho_cfd, "b.-", label=r"$\rho$")
        ax[1].set_xlabel(r"$y$")
        ax[1].set_ylabel(r"$\rho - \rho_0 ( kg/m^3 )$")
        
             
        ax[0].legend(prop={'size':8},loc="lower left")
        ax[1].legend(prop={'size':8},loc="lower left")     
                
        # Save
        fig.set_size_inches(x_full, x_full/2)
        fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
        fig.savefig("pRho.jpeg", dpi=600,bbox_inches='tight')
        
        
        ####################################
        ## Clean
        ####################################
        plt.close('all')
        
        
# ****************************************************************************
def run():
    solver = LDC()
    solver.solve()
    print "time", round(time() - solver.start,2), "s"

# # ****************************************************************************
if __name__ == "__main__":
    print "START"
    run()
    print "END"
    
    
    
    
    
    
