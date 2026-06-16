"""Analytic Cerfon--Freidberg Solov'ev equilibrium solver.

Implements the ``Device`` class, which solves the seven boundary-coefficient
linear system for a given tokamak geometry (epsilon, kappa, delta) and the
profile constant A, following Cerfon & Freidberg (2010). Used by
scripts/generate_data.py to produce the training/evaluation datasets.

Requires JAX (autograd of the basis functions) and SymPy (symbolic linear
solve). Plotting helpers use matplotlib and are optional.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # JAX ops require jnp (NOT np) for autograd
from jax import grad      # autograd
import sympy as sp        # symbolic differentiation / linear solve

class Device():

    def __init__(self, pressure, epsilon, kappa, delta, device_name):

        self.pressure = pressure
        self.epsilon = epsilon # inverse aspect ratio
        self.kappa = kappa # elongation
        self.delta = delta # triangularity

        self.device_name = device_name

        self.r_o = 1.0 + self.epsilon # outer point r value
        self.z_o = 0.0 # outer point z value
        self.r_i = 1.0 - self.epsilon # inner point r value
        self.z_i = 0.0 # inner point z value
        self.r_h = 1.0 - self.delta * self.epsilon # high point r value
        self.z_h = self.kappa * self.epsilon # high point z value

        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7 = self.get_analytical_flux_coeff().args[0]

    def get_parametric_boundary_RZ_values(self, tau_array = np.arange(0, 2 * np.pi + 0.1, 0.1)):

        '''
        obtain (R, Z) using parametric equations --- see equation (8)
        '''
        r_value_array = 1.0 + self.epsilon * np.cos(tau_array + np.arcsin(self.delta) * np.sin(tau_array))
        z_value_array = self.epsilon * self.kappa * np.sin(tau_array)

        return r_value_array, z_value_array

    def plot_parametric_boundary(self):
        '''
        plot boundary using parametric equations --- see equation (8)
        '''

        r_values, z_values = self.get_parametric_boundary_RZ_values()

        fig, ax = plt.subplots(1, 1, figsize=(5,8))
        ax.plot(r_values, z_values)
        ax.set_xlabel(r'$R$')
        ax.set_ylabel(r'$Z$')
        ax.set_title(f"{self.device_name} parametric boundary plot", fontsize=16)
        ax.set_yticks(np.arange(-1.5, 2, 0.5))
        ax.set_xticks(range(0, 3, 1))
        # plt.grid()
        plt.show()

    # particular solution
    def psiP(self, r, z):
        return self.pressure * (r**4)/8.0 + (1.0 - self.pressure) * ((r**2)/2.0) * (jnp.log(r)) # use jnp.log and NOT np.log

    def psiH(self, r, z):
        homo_soln = (
                        self.c1 * self.psi_1(r, z) + self.c2 * self.psi_2(r, z) + self.c3 * self.psi_3(r, z)
                        + self.c4 * self.psi_4(r, z) + self.c5 * self.psi_5(r, z) + self.c6 * self.psi_6(r, z)
                        + self.c7 * self.psi_7(r, z)
                     )
        return homo_soln
    
    # define the seven up-down symmetric functions
    def psi_1(self, r, z):
        return 1.0

    def psi_2(self, r, z):
        return r**2

    def psi_3(self, r, z):
        return z**2 - (r**2) * jnp.log(r)

    def psi_4(self,r, z):
        return r**4 - 4.0 * (r**2) * (z**2)

    def psi_5(self,r, z):
        return 2.0 * (z**4) - 9.0 * (z**2) * (r**2) - (12.0 * (z**2) * (r**2) - 3.0 * (r**4)) * jnp.log(r)

    def psi_6(self,r, z):
        return r**6 - 12.0 * (r**4) * (z**2) + 8.0 * (r**2) * (z**4)

    def psi_7(self,r, z):
        return 8.0 * (z**6) - 140.0 * (z**4) * (r**2) + 75.0 * (z**2) * (r**4) - (120.0 * (z**4) * (r**2) - 180.0 * (z**2) * (r**4) + 15.0 * (r**6)) * jnp.log(r)


    # calculate the coefficients of the analytical solutions
    def get_analytical_flux_coeff(self):
        '''
        Solves for the seven coefficients c1 to c7 in the analytical solution
        of the SCALED poloidal flux (Cerfon solution) --- see equation (10)
        '''

        # calculate curvature coefficients
        N1 = - (1.0 + jnp.arcsin(self.delta)) ** 2 / (self.epsilon * self.kappa ** 2)
        N2 = (1.0 - jnp.arcsin(self.delta)) ** 2 / (self.epsilon * self.kappa ** 2)
        N3 = - self.kappa / (self.epsilon * jnp.cos(np.arcsin(self.delta)) ** 2)

        # derivatives of particular solution
        psiP_r = grad(self.psiP, argnums = 0)
        psiP_rr = grad(psiP_r, argnums = 0)
        psiP_z = lambda r, z: 0.0
        psiP_zz = lambda r, z: 0.0

        # homogeneous solution (up-down symmetric functions) and their derivatives
        # psi_1 = lambda r, z: 1.0
        psi_1_r = lambda r, z: 0.0
        psi_1_rr = lambda r, z: 0.0
        psi_1_z = lambda r, z: 0.0
        psi_1_zz = lambda r, z: 0.0

        # psi_2 = lambda r, z: r**2
        psi_2_r = grad(self.psi_2, argnums = 0)
        psi_2_rr = grad(psi_2_r, argnums = 0)
        psi_2_z = lambda r, z: 0.0
        psi_2_zz = lambda r, z: 0.0

        psi_3_r = grad(self.psi_3, argnums = 0)
        psi_3_rr = grad(psi_3_r, argnums = 0)
        psi_3_z = grad(self.psi_3, argnums = 1)
        psi_3_zz = grad(psi_3_z, argnums = 1)

        psi_4_r = grad(self.psi_4, argnums = 0)
        psi_4_rr = grad(psi_4_r, argnums = 0)
        psi_4_z = grad(self.psi_4, argnums = 1)
        psi_4_zz = grad(psi_4_z, argnums = 1)

        psi_5_r = grad(self.psi_5, argnums = 0)
        psi_5_rr = grad(psi_5_r, argnums = 0)
        psi_5_z = grad(self.psi_5, argnums = 1)
        psi_5_zz = grad(psi_5_z, argnums = 1)

        psi_6_r = grad(self.psi_6, argnums = 0)
        psi_6_rr = grad(psi_6_r, argnums = 0)
        psi_6_z = grad(self.psi_6, argnums = 1)
        psi_6_zz = grad(psi_6_z, argnums = 1)

        psi_7_r = grad(self.psi_7, argnums = 0)
        psi_7_rr = grad(psi_7_r, argnums = 0)
        psi_7_z = grad(self.psi_7, argnums = 1)
        psi_7_zz = grad(psi_7_z, argnums = 1)


        #-------------------------SYSTEM OF EQUATIONS---------------------------
        # define algebraic symbols
        c1, c2, c3, c4, c5, c6, c7 = sp.symbols('c1 c2 c3 c4 c5 c6 c7')

        # equation 1 (outer point flux)
        psi_full_outer = (
                            self.psiP(self.r_o, self.z_o)
                            + c1 * self.psi_1(self.r_o, self.z_o) + c2 * self.psi_2(self.r_o, self.z_o)
                            + c3 * self.psi_3(self.r_o, self.z_o) + c4 * self.psi_4(self.r_o, self.z_o)
                            + c5 * self.psi_5(self.r_o, self.z_o) + c6 * self.psi_6(self.r_o, self.z_o)
                            + c7 * self.psi_7(self.r_o, self.z_o)
                            )

        eq1 = sp.Eq(psi_full_outer, 0)

        # equation 2 (outer point curvature)
        psi_full_outer_zz = (
                            psiP_zz(self.r_o, self.z_o)
                            + c1 * psi_1_zz(self.r_o, self.z_o) + c2 * psi_2_zz(self.r_o, self.z_o)
                            + c3 * psi_3_zz(self.r_o, self.z_o) + c4 * psi_4_zz(self.r_o, self.z_o)
                            + c5 * psi_5_zz(self.r_o, self.z_o) + c6 * psi_6_zz(self.r_o, self.z_o)
                            + c7 * psi_7_zz(self.r_o, self.z_o)
                            )

        psi_full_outer_r = (
                            psiP_r(self.r_o, self.z_o)
                            + c1 * psi_1_r(self.r_o, self.z_o) + c2 * psi_2_r(self.r_o, self.z_o)
                            + c3 * psi_3_r(self.r_o, self.z_o) + c4 * psi_4_r(self.r_o, self.z_o)
                            + c5 * psi_5_r(self.r_o, self.z_o) + c6 * psi_6_r(self.r_o, self.z_o)
                            + c7 * psi_7_r(self.r_o, self.z_o)
                            )

        eq2 = sp.Eq(psi_full_outer_zz + N1 * psi_full_outer_r, 0)

        # equation 3 (inner point flux)
        psi_full_inner = (
                            self.psiP(self.r_i, self.z_i)
                            + c1 * self.psi_1(self.r_i, self.z_i) + c2 * self.psi_2(self.r_i, self.z_i)
                            + c3 * self.psi_3(self.r_i, self.z_i) + c4 * self.psi_4(self.r_i, self.z_i)
                            + c5 * self.psi_5(self.r_i, self.z_i) + c6 * self.psi_6(self.r_i, self.z_i)
                            + c7 * self.psi_7(self.r_i, self.z_i)
                            )

        eq3 = sp.Eq(psi_full_inner, 0)

        # equation 4 (inner point curvature)
        psi_full_inner_zz = (
                            psiP_zz(self.r_i, self.z_i)
                            + c1 * psi_1_zz(self.r_i, self.z_i) + c2 * psi_2_zz(self.r_i, self.z_i)
                            + c3 * psi_3_zz(self.r_i, self.z_i) + c4 * psi_4_zz(self.r_i, self.z_i)
                            + c5 * psi_5_zz(self.r_i, self.z_i) + c6 * psi_6_zz(self.r_i, self.z_i)
                            + c7 * psi_7_zz(self.r_i, self.z_i)
                            )

        psi_full_inner_r = (
                            psiP_r(self.r_i, self.z_i)
                            + c1 * psi_1_r(self.r_i, self.z_i) + c2 * psi_2_r(self.r_i, self.z_i)
                            + c3 * psi_3_r(self.r_i, self.z_i) + c4 * psi_4_r(self.r_i, self.z_i)
                            + c5 * psi_5_r(self.r_i, self.z_i) + c6 * psi_6_r(self.r_i, self.z_i)
                            + c7 * psi_7_r(self.r_i, self.z_i)
                            )

        eq4 = sp.Eq(psi_full_inner_zz + N2 * psi_full_inner_r, 0)

        # equation 5 (high point flux)
        psi_full_high = (
                            self.psiP(self.r_h, self.z_h)
                            + c1 * self.psi_1(self.r_h, self.z_h) + c2 * self.psi_2(self.r_h, self.z_h)
                            + c3 * self.psi_3(self.r_h, self.z_h) + c4 * self.psi_4(self.r_h, self.z_h)
                            + c5 * self.psi_5(self.r_h, self.z_h) + c6 * self.psi_6(self.r_h, self.z_h)
                            + c7 * self.psi_7(self.r_h, self.z_h)
                            )

        eq5 = sp.Eq(psi_full_high, 0)

        # equation 6 (high point slope)
        psi_full_high_r = (
                            psiP_r(self.r_h, self.z_h)
                            + c1 * psi_1_r(self.r_h, self.z_h) + c2 * psi_2_r(self.r_h, self.z_h)
                            + c3 * psi_3_r(self.r_h, self.z_h) + c4 * psi_4_r(self.r_h, self.z_h)
                            + c5 * psi_5_r(self.r_h, self.z_h) + c6 * psi_6_r(self.r_h, self.z_h)
                            + c7 * psi_7_r(self.r_h, self.z_h)
                            )

        eq6 = sp.Eq(psi_full_high_r, 0)

        # equation 7 (high point curvature)
        psi_full_high_rr = (
                            psiP_rr(self.r_h, self.z_h)
                            + c1 * psi_1_rr(self.r_h, self.z_h) + c2 * psi_2_rr(self.r_h, self.z_h)
                            + c3 * psi_3_rr(self.r_h, self.z_h) + c4 * psi_4_rr(self.r_h, self.z_h)
                            + c5 * psi_5_rr(self.r_h, self.z_h) + c6 * psi_6_rr(self.r_h, self.z_h)
                            + c7 * psi_7_rr(self.r_h, self.z_h)
                            )

        psi_full_high_z = (
                            psiP_z(self.r_h, self.z_h)
                            + c1 * psi_1_z(self.r_h, self.z_h) + c2 * psi_2_z(self.r_h, self.z_h)
                            + c3 * psi_3_z(self.r_h, self.z_h) + c4 * psi_4_z(self.r_h, self.z_h)
                            + c5 * psi_5_z(self.r_h, self.z_h) + c6 * psi_6_z(self.r_h, self.z_h)
                            + c7 * psi_7_z(self.r_h, self.z_h)
                            )

        eq7 = sp.Eq(psi_full_high_rr + N3 * psi_full_high_z, 0)

        # calculate coefficients
        solution = sp.linsolve([eq1, eq2, eq3, eq4, eq5, eq6, eq7], c1, c2, c3, c4, c5, c6, c7)

        return solution

    def compute_full_flux(self, r, z, c1, c2, c3, c4, c5, c6, c7):

        particular_soln = self.psiP(r, z)
        homo_soln = (
                        c1 * self.psi_1(r, z) + c2 * self.psi_2(r, z) + c3 * self.psi_3(r, z)
                        + c4 * self.psi_4(r, z) + c5 * self.psi_5(r, z) + c6 * self.psi_6(r, z)
                        + c7 * self.psi_7(r, z)
                     )

        return particular_soln + homo_soln
    
    
    
    def calculate_analytical_psi(self, r_val, z_val):
        return self.compute_full_flux(r_val, z_val, self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7)


    def analytical_flux_contour_plot(self, r_array, z_array):

        # coefficients
        # c1, c2, c3, c4, c5, c6, c7 = self.get_analytical_flux_coeff().args[0]
        # print(c1, c2, c3, c4, c5, c6, c7)

        # set up mesh grid
        x, y = np.meshgrid(r_array, z_array)

        # calculate analytical psi
        psi_analytic = np.ones((len(r_array), len(z_array)))

        for r_idx in range(len(r_array)):
            for z_idx in range(len(z_array)):
                psi_analytic[z_idx, r_idx] = self.compute_full_flux(r_array[r_idx], z_array[z_idx], self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7)

        # contour plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 8), dpi = 100)
        contour_levels = np.linspace(-0.225,0.0,11)
        contour_plot = ax.contour(x, y, psi_analytic, cmap='viridis', levels = contour_levels)
        ax.set_xlabel(r'$R$', fontsize=12)
        ax.set_ylabel(r'$Z$', fontsize=12)
        ax.set_title(f"{self.device_name} flux surface (analytical solution)", fontsize=16, pad = 20)
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.set_xticks(range(0, 3, 1))
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        cbar = plt.colorbar(contour_plot)
        cbar.ax.tick_params(labelsize=12)
        plt.grid()
        plt.show()

    def compare_with_analytical_soln(self, r_array, z_array, model_output_psi_grid):

        # coefficients
        # c1, c2, c3, c4, c5, c6, c7 = self.get_analytical_flux_coeff().args[0]
        # print(c1, c2, c3, c4, c5, c6, c7)

        # set up mesh grid
        x, y = np.meshgrid(r_array, z_array)

        # calculate analytical psi
        psi_analytic = np.ones((len(r_array), len(z_array)))

        for r_idx in range(len(r_array)):
            for z_idx in range(len(z_array)):
                psi_analytic[z_idx, r_idx] = self.compute_full_flux(r_array[r_idx], z_array[z_idx], self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7)

        # contour plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        contour_levels = np.linspace(-0.225,0.0,11)
        contour_plot = ax[0].contour(x, y, psi_analytic, cmap='viridis', levels = contour_levels)
        contour_plot_model = ax[0].contour(x, y, model_output_psi_grid, cmap='viridis', levels = contour_levels, linestyles = 'dashed')
        cbar1 = fig.colorbar(contour_plot_model, ax=ax[0])
        cbar1.ax.tick_params(labelsize=12)
        ax[0].set_xlabel(r'$R$', fontsize = 12)
        ax[0].set_ylabel(r'$Z$', fontsize = 12)
        ax[0].set_title(f"{self.device_name} analytical & model comparison", fontsize=16, pad = 20)
        ax[0].set_yticks(np.arange(-2, 3, 1))
        ax[0].set_xticks(range(0, 3, 1))
        ax[0].tick_params(axis='x', labelsize=12)
        ax[0].tick_params(axis='y', labelsize=12)
        ax[0].grid

        contour_plot_model2 = ax[1].contour(x, y, model_output_psi_grid, cmap='viridis', levels = contour_levels, linestyles = 'dashed')
        cbar2 = fig.colorbar(contour_plot_model2, ax=ax[1])
        cbar2.ax.tick_params(labelsize=12)
        ax[1].set_xlabel(r'$R$', fontsize = 12)
        ax[1].set_ylabel(r'$Z$', fontsize = 12)
        ax[1].set_title(f"{self.device_name} model only", fontsize=16, pad = 20)
        ax[1].set_yticks(np.arange(-2, 3, 1))
        ax[1].set_xticks(range(0, 3, 1))
        ax[1].tick_params(axis='x', labelsize=12)
        ax[1].tick_params(axis='y', labelsize=12)
        ax[1].grid

        plt.show()