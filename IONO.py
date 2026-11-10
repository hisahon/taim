# from numpy import *
from numpy import sin, cos
from scipy import *
from SIM_VOL import *
from AMBIENT import *

import os
fname = os.path.basename(__file__)
print(fname)

global i_o,i_amb,i_pot,i_inert,n_sor

i_inert=1;i_amb=0;i_pot=0;i_o=0;n_sor=3

q_charge=1.6e-19;b_c=1.38e-23;eps_o=8.854e-12;m_i=1.67e-27;z_i=16.
mag_o=0.25e-04;
#=============================================================================#

def ndf(x,mu,sig):
    return exp(-power((x-mu)/sig,2.)/2)
    

#=============================================================================#
def mobility(np_t):        #(mu_p,-mu_h//mu_h,mu_p)
    
##    gr_p=up_tot.max()*abs(gradient(log(np_t))[0]/dp_i)
##    for j in range (0,nf):
##        for k in range (0,nq):
##            gr_p[:,j,k]=gr_p[:,iphi_m,iq_m]

    gr_p=0.;tau=dt/1.
    nu_inert=i_inert*(1./tau+2.*gr_p);
    nu_eff=nu_i+nu_inert
    
    kappa=gyro_i/nu_eff
    mu_p_i=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_i=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_i=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    nu_eff=nu_e+nu_inert
    
    kappa=gyro_e/nu_eff
    mu_p_e=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_e=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_e=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    return (mu_p_i,mu_h_i,mu_o_i,mu_p_e,mu_h_e,mu_o_e)
    
def conductivity(np_t):        #(s_p,-s_h//s_h,s_p)
    f=mobility(np_t)
    mu_p_i=f[0];mu_h_i=f[1];mu_o_i=f[2];
    mu_p_e=f[3];mu_h_e=f[4];mu_o_e=f[5];    

    sigma_p=q_charge*np_t*(mu_p_i-mu_p_e)
    sigma_h=-q_charge*np_t*(mu_h_i-mu_h_e)
    sigma_o=q_charge*np_t*(mu_o_i-mu_o_e)
     
    return (sigma_p,sigma_h,sigma_o)


def calculate_magnetic_field_components(theta, r):
    """
    Calculate the magnetic field components in spherical coordinates (radial, polar, and azimuthal)
    for a dipole field.

    Parameters:
    -----------
    θ : ndarray
        3D array representing the polar angle (in radians) at each grid point.
    r : ndarray
        3D array representing the radial distance at each grid point.
    np_start : int
        Start index for the computation domain.
    np_end : int
        End index for the computation domain.

    Returns:
    --------
    Bᵣ : ndarray
        Radial component of the magnetic field.
    B_θ : ndarray
        Polar (θ) component of the magnetic field.
    B_φ : float
        Azimuthal (φ) component of the magnetic field (always zero for axial symmetry).
    
    Theory:
    -------
    The magnetic field of a dipole in spherical coordinates is described by:
    1. Radial component (Bᵣ):
       Bᵣ = -2 * M * sin(θ) / r³
       - This component varies with the sine of the polar angle (θ) and decreases with the cube of the radial distance (r).
    2. Polar component (B_θ):
       B_θ = M * cos(θ) / r³
       - This component varies with the cosine of the polar angle (θ) and decreases with the cube of the radial distance (r).
    3. Azimuthal component (B_φ):
       B_φ = 0
       - Due to axial symmetry, there is no variation along the azimuthal direction (φ), making this component zero.

    M is the magnetic moment of the dipole.
    """

    # Magnetic moment of the dipole, in T·m³
    b_o = 8.e+15  

    # Radial component of the magnetic field
    b_r = -2 * b_o * sin(theta[np-np_i, :, :]) / r[np-np_i, :, :]**3  

    # Polar component of the magnetic field
    b_theta = b_o * cos(theta[np-np_i, :, :]) / r[np-np_i, :, :]**3  

    # Azimuthal component of the magnetic field
    b_phi = 0.  # Assuming axial symmetry, this component is zero.
    
    # import matplotlib.pyplot as plt
    # # plt.figure(figsize=(8, 5))
    # # plt.pcolormesh(lon, lat, b_phi.T, cmap='Blues_r')
    # # # plt.contourf(lon, lat, Be[0].T, cmap='Blues', levels = 5)
    # # plt.colorbar(label='Field Intensity East')
    # # plt.ylabel('Latitude')
    # # plt.xlabel('Longitude')
    # # plt.title('Magnetic Field East Direction')
    
    # plt.figure(figsize=(8, 5))
    # plt.pcolormesh(lon, lat, b_theta.T, cmap='Greens_r')
    # # plt.contourf(lon, lat, Be[0].T, cmap='Blues', levels = 5)
    # plt.colorbar(label='Field Intensity North')
    # plt.ylabel('Latitude')
    # plt.xlabel('Longitude')
    # plt.title('Magnetic Field North Direction')
    
    # plt.figure(figsize=(8, 5))
    # plt.pcolormesh(lon, lat, b_r.T, cmap='Greys_r')
    # # plt.contourf(lon, lat, Be[0].T, cmap='Blues', levels = 5)
    # plt.colorbar(label='Field Intensity Up')
    # plt.ylabel('Latitude')
    # plt.xlabel('Longitude')
    # plt.title('Magnetic Field Up Direction')
    # plt.show()
    
    

    return b_r, b_theta, b_phi


def calculate_electric_field_components(wp_g, wf_g, wq_g):
    """
    Calculate the components of the electric field in spherical coordinates based on 
    the angular velocity and magnetic field vectors.

    Theory:
    -------
    The electric field (E) in a rotating system can be derived using the Lorentz force 
    equation: F = q(E + v x B), where E is the electric field, v is the velocity of the 
    charged particle, and B is the magnetic field. In this context, the velocity v is 
    replaced by angular velocity components in a rotating reference frame:
        v = w x r
    where w is the angular velocity vector and r is the position vector.

    In spherical coordinates, the components of the electric field are obtained 
    by computing the cross-product of the angular velocity vector (w) and the magnetic 
    field vector (B), E = - w x B. For a system with angular velocity components (wp_g, wf_g, wq_g) 
    and magnetic field components (b_r, b_theta, b_phi), the resulting electric field 
    components are derived as follows:

    1. The polar component (e_p):
       Derived from the interaction of the azimuthal angular velocity (wf_g) with 
       the polar magnetic field (b_theta), and the radial angular velocity (wq_g) 
       with the azimuthal magnetic field (b_phi):
           e_p = wf_g * b_theta - wq_g * b_phi

    2. The azimuthal component (e_f):
       Derived from the interaction of the radial angular velocity (wq_g) with 
       the radial magnetic field (b_r), and the polar angular velocity (wp_g) 
       with the polar magnetic field (b_theta):
           e_f = wq_g * b_r - wp_g * b_theta

    3. The radial component (e_q):
       Derived from the interaction of the polar angular velocity (wp_g) with 
       the azimuthal magnetic field (b_phi), and the azimuthal angular velocity (wf_g) 
       with the radial magnetic field (b_r):
           e_q = wp_g * b_phi - wf_g * b_r

    Parameters:
    -----------
    wp_g : float or np.ndarray
        Angular velocity in the polar direction (rad/s).
    wf_g : float or np.ndarray
        Angular velocity in the azimuthal direction (rad/s).
    wq_g : float or np.ndarray
        Angular velocity in the radial direction (rad/s).
    b_r : float or np.ndarray
        Radial component of the magnetic field (T).
    b_theta : float or np.ndarray
        Polar component of the magnetic field (T).
    b_phi : float or np.ndarray
        Azimuthal component of the magnetic field (T).

    Returns:
    --------
    e_p : float or np.ndarray
        Polar component of the electric field (V/m).
    e_f : float or np.ndarray
        Azimuthal component of the electric field (V/m).
    e_q : float or np.ndarray
        Radial component of the electric field (V/m).
    """

    # B_components = calculate_magnetic_field_components(theta, r)
    # b_r, b_theta, b_phi = B_components
    
    # angle_I, Be, Bn, Bu = B_igrf
    _, b_phi, b_theta, b_r = B_igrf


    # Calculate the polar component of the electric field
    # This is determined by the azimuthal angular velocity interacting with the polar magnetic field 
    # and the radial angular velocity interacting with the azimuthal magnetic field.
    e_p = wf_g * b_theta - wq_g * b_phi  # Polar component (V/m)

    # Calculate the azimuthal component of the electric field
    # This is determined by the radial angular velocity interacting with the radial magnetic field 
    # and the polar angular velocity interacting with the polar magnetic field.
    e_f = wq_g * b_r - wp_g * b_theta  # Azimuthal component (V/m)

    # Calculate the radial component of the electric field
    # This is determined by the polar angular velocity interacting with the azimuthal magnetic field 
    # and the azimuthal angular velocity interacting with the radial magnetic field.
    e_q = wp_g * b_phi - wf_g * b_r  # Radial component (V/m)

    # Return the components of the electric field
    return e_p, e_f, e_q


def mobility_tensor(gyro_i, nu_i, mag_o, angle_I):
    """
    Function to calculate the components of the mobility tensor based on the gyrofrequency (gyro_i),
    the mean collision frequency (nu_i), magnetic field magnitude (mag_o), and the angle (angle_I).

    Full mobility tensor for reference:
        μ = [
            [μ11, μ12, μ13],
            [μ21, μ22, μ23],
            [μ31, μ32, μ33]
            ]
        
        μ = [
            [μ_p, -μ_p * κ * cos(θ_I), -μ_p * κ * sin(θ_I)],
            [μ_p * κ * cos(θ_I), μ_p * (1 + κ**2 * sin(θ_I)**2), -μ_p * κ**2 * cos(θ_I) * sin(θ_I)],
            [μ_p * κ * sin(θ_I), -μ_p * κ**2 * cos(θ_I) * sin(θ_I), μ_p * (1 + κ**2 * cos(θ_I)**2)]
        ]
    
    Parameters:
    - gyro_i: Gyrofrequency (scalar)
    - nu_i: Array of collision frequencies
    - mag_o: Magnetic field magnitude (scalar)
    - angle_I: Angle (scalar, in radians)
    
    Returns:
    - mu_11, mu_12, mu_13, mu_21, mu_22, mu_23, mu_31, mu_32, mu_33
    """
    
    
    # Calculation of kappa: ratio between gyrofrequency and mean collision frequency
    kappa = gyro_i / nu_i.mean()
    
    # Base mobility term
    mu_p = kappa / (mag_o * (1 + kappa**2))  # Base mobility term
    
    # Diagonal elements of the mobility tensor (μ)
    mu_11 = mu_p  # Polar direction (μ11)
    mu_22 = mu_11 * (1. + kappa**2 * sin(angle_I)**2)  # Azimuthal direction (μ22)
    mu_33 = mu_11 * (1. + kappa**2 * cos(angle_I)**2)  # Radial direction (μ33)
    
    # Off-diagonal elements of the mobility tensor (μ)
    mu_12 = -mu_11 * kappa * cos(angle_I)  # Coupling between polar and azimuthal directions (μ12)
    mu_13 = -mu_11 * kappa * sin(angle_I)  # Coupling between polar and radial directions (μ13)
    mu_23 = -mu_11 * kappa**2 * cos(angle_I) * sin(angle_I)  # Coupling between azimuthal and radial directions (μ23)
    
    # Symmetric elements of the mobility tensor (μ)
    mu_21 = -mu_12  # Symmetric term for polar-azimuthal coupling (μ21)
    mu_31 = -mu_13  # Symmetric term for polar-radial coupling (μ31)
    mu_32 = mu_23  # Symmetric term for azimuthal-radial coupling (μ32)
    

    return mu_11, mu_12, mu_13, mu_21, mu_22, mu_23, mu_31, mu_32, mu_33


   
def den_evolve2(np_o,wp_g,wf_g,wq_g):
    
    '''
    The Acoustic Gravity Wave Induced Disturbances in the Equatorial Ionosphere
    ISBN 978-94-007-0325-4 978-94-007-0326-1
    https://link.springer.com/10.1007/978-94-007-0326-1_10
    
    '''
    angle_I, _, _, _ = B_igrf
    
    #Calculation the electric Field Components
    E_pfq = calculate_electric_field_components(wp_g, wf_g, wq_g)
    e_p, e_f, e_q = E_pfq
    
    #Calculating the mobility Tensor    
    mu_tensor = mobility_tensor(gyro_i, nu_i, mag_o, angle_I)
    mu_11, mu_12, mu_13, mu_21, mu_22, mu_23, mu_31, mu_32, mu_33 = mu_tensor
        
    # Calculate the total velocity components (or other relevant quantities) in each direction using the mobility tensor
    up_tot = mu_11 * e_p + mu_12 * e_f + mu_13 * e_q  # Velocity component in the polar direction
    uf_tot = mu_22 * e_f + mu_21 * e_p + mu_23 * e_q  # Velocity component in the azimuthal direction
    uq_tot = mu_33 * e_q + mu_31 * e_p + mu_32 * e_f  # Velocity component in the radial direction
    # Uncommented section to apply drift calculations (if needed)
    # f_d = drift(e_p, e_f, e_q)
    ## up_tot = f_d[0]  # Polar velocity from drift
    ## uf_tot = f_d[1]  # Azimuthal velocity from drift
    ## uq_tot = f_d[2]  # Radial velocity from drift
    
    # Initialize particle density (np_g) as a reference value
    np_g = np_o

    for i_sor in range(0, n_sor):
        # Calculate flux components
        flux_p = np_o * up_tot
        flux_f = np_o * uf_tot
        flux_q = np_o * uq_tot
        
        # Compute gradients of flux components
        gr_p = gradient(flux_p, axis=0, edge_order=2)
        gr_f = gradient(flux_f, axis=1, edge_order=2)
        gr_q = gradient(flux_q, axis=2, edge_order=2)
        
        # Divergence of flux for original density (np_o)
        div_flux_o = gr_p / dp_i + gr_f / df_i + gr_q / dq_i
        
        # Recompute flux for updated density (np_g)
        flux_p = np_g * up_tot
        flux_f = np_g * uf_tot
        flux_q = np_g * uq_tot
        
        gr_p = gradient(flux_p, axis=0, edge_order=2)
        gr_f = gradient(flux_f, axis=1, edge_order=2)
        gr_q = gradient(flux_q, axis=2, edge_order=2)
        
        # Divergence of flux for updated density (np_g)
        div_flux = gr_p / dp_i + gr_f / df_i + gr_q / dq_i
        
#            gr_p=gradient(log(np_o))
#            div_flux_o=np_o*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i+0*i_o*uq_tot*gr_p[2]/dq_i)
#            gr_p=gradient(log(np_g))
#            div_flux=np_g*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i+0*i_o*uq_tot*gr_p[2]/dq_i)

        
        # Updated density using time-stepping
        np_t = np_o - 6. * dt * (div_flux + div_flux_o) / 2.
        np_g = (np_g + np_t) / 2.
        
        delta_sor = (abs(np_g - np_t) / np_t).max()
        if delta_sor <1.e-05:
            # print ('CONVERGENCE OF CONTINUITY',i_sor)
            break
        
    return (np_t)

def den_evolve(np_o,wp_g,wf_g,wq_g):
    
    '''
    The Acoustic Gravity Wave Induced Disturbances in the Equatorial Ionosphere
    ISBN 978-94-007-0325-4 978-94-007-0326-1
    https://link.springer.com/10.1007/978-94-007-0326-1_10
    '''
    
##    e_p=-wf_g*mag_o;e_f=wp_g*mag_o*cos(angle_I)+wq_g*mag_o*sin(angle_I);
##    e_q=-wf_g*mag_o*sin(angle_I)+wq_g*mag_o*sin(angle_I);
    
    b_o=8.e+15         #Tm^3
    b_r=-2*b_o*sin(theta[np-np_i:,:,:])/r[np-np_i:,:,:]**3.;
    b_theta=b_o*cos(theta[np-np_i:,:,:])/r[np-np_i:,:,:]**3.
    b_phi=0.
    
    e_p=wf_g*b_theta-wq_g*b_phi
    e_f=wq_g*b_r-wp_g*b_theta
    e_q=wp_g*b_phi-wf_g*b_r
    
    kappa=gyro_i/nu_i.mean()
    mu_p=(kappa/(mag_o*(1+kappa**2.)))
    mu_11=mu_p;mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);
    mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    #vec u
    up_tot=mu_11*e_p+mu_12*e_f+mu_13*e_q
    uf_tot=mu_22*e_f+mu_21*e_p+mu_23*e_q
    uq_tot=mu_33*e_q+mu_31*e_p+mu_32*e_f
##    f_d=drift(e_p,e_f,e_q)
##    up_tot=f_d[0];uf_tot=f_d[1];uq_tot=f_d[2]
##    up_tot=wp_g;uf_tot=wf_g;
##    uq_tot=wq_g
    np_g=np_o
    for i_sor in range (0,n_sor):
 
        flux_p=np_o*up_tot;flux_f=np_o*uf_tot;flux_q=np_o*uq_tot;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)   
        div_flux_o=gr_p[0]/dp_i+gr_f[1]/df_i+gr_q[2]/dq_i
        
        flux_p=np_g*up_tot;flux_f=np_g*uf_tot;flux_q=np_g*uq_tot;
        gr_p=gradient(flux_p);gr_f=gradient(flux_f);gr_q=gradient(flux_q)   
        div_flux=gr_p[0]/dp_i+gr_f[1]/df_i+gr_q[2]/dq_i

#            gr_p=gradient(log(np_o))
#            div_flux_o=np_o*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i+0*i_o*uq_tot*gr_p[2]/dq_i)
#            gr_p=gradient(log(np_g))
#            div_flux=np_g*(up_tot*gr_p[0]/dp_i+uf_tot*gr_p[1]/df_i+0*i_o*uq_tot*gr_p[2]/dq_i)
        
        np_t=np_o-6.*dt*(div_flux+div_flux_o)/2.
        delta_sor=(abs(np_g-np_t)/np_t).max()
        np_g=(np_g+np_t)/2.
        
        if delta_sor <1.e-05:
            # If the convergenge continuity was archived break the loop
            # print ('CONVERGENCE OF CONTINUITY',i_sor)
            break
        
    return (np_t)

#=============================================================================#

def drift(e_p,e_f,e_q):
    
    f=mobility(np_amb)
    
    kappa=f[2]*mag_o
    mu_11=f[0];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);
    mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    up_i=mu_11*e_p+mu_12*e_f+mu_13*e_q
    uf_i=mu_22*e_f+mu_21*e_p+mu_23*e_qangle_I
    uq_i=mu_33*e_q+mu_31*e_p+mu_32*e_f


    kappa=f[5]*mag_o
    mu_11=f[3];mu_12=-mu_11*kappa*cos(angle_I);mu_13=-mu_11*kappa*sin(angle_I)
    mu_21=-mu_12;mu_22=mu_11*(1.+kappa**2.*sin(angle_I)**2.);
    mu_23=-mu_11*kappa**2.*cos(angle_I)*sin(angle_I)   
    mu_31=-mu_13;mu_32=mu_23;mu_33=mu_11*(1.+kappa**2.*cos(angle_I)**2.)
    
    up_e=mu_11*e_p+mu_12*e_f+mu_13*e_q
    uf_e=mu_22*e_f+mu_21*e_p+mu_23*e_q
    uq_e=mu_33*e_q+mu_31*e_p+mu_32*e_f
    
    return (up_i,uf_i,uq_i,up_e,uf_e,uq_e)
    
#=============================================================================#    

def poisson(t_e,wp_g,wf_g,wq_g,np_t,pot):
    global up_tot,uf_tot,uq_tot
    up_tot=0*np_t;
#===
    f=conductivity(np_t)
    sigma_p=f[0];sigma_h=f[1];sigma_o=f[2];
    
    s_11=sigma_p;s_12=-sigma_h*cos(angle_I);s_13=-sigma_h*sin(angle_I)
    s_21=-s_12;
    s_22=sigma_p*cos(angle_I)**2.+sigma_o*sin(angle_I)**2.;
    s_23=(sigma_p-sigma_o)*sin(angle_I)*cos(angle_I)
    s_31=-s_13;s_32=s_23;
    s_33=sigma_o*cos(angle_I)**2.+sigma_p*sin(angle_I)**2.    
    
    s_11=s_22
       
#===WIND DRIVEN DRIFs/CURRENTs
    
    e_p=wf_g*mag_o;e_f=-wp_g*mag_o*cos(angle_I);e_q=wq_g*mag_o#*sin(angle_I)
    
    f_d=drift(e_p,e_f,e_q)
    up_agw_i=f_d[0];uf_agw_i=f_d[1];uq_agw_i=f_d[2]
    up_agw_e=f_d[3];uf_agw_e=f_d[4];uq_agw_e=f_d[5]

    cur_wind_p=np_t*q_charge*(up_agw_i-up_agw_e)
    cur_wind_f=np_t*q_charge*(uf_agw_i-uf_agw_e)
    cur_wind_q=np_t*q_charge*(uq_agw_i-uq_agw_e)   
    
    
    div_cur_wind=gradient(cur_wind_p)[0]/dp_i+gradient(cur_wind_f)[1]/df_i+gradient(cur_wind_q)[2]/dq_i
    source_wind=-div_cur_wind/sigma_p
    
#===GRAVITY DRIVEN DRIFTS/CURRENTS
    
    gr=0*i_amb*9.8;gr_p=-gr;gr_f=0;gr_q=0
    fac=mag_o/gyro_i
    e_p=fac*gr_p;e_f=fac*gr_f;e_q=fac*gr_q;
    
    f_d=drift(e_p,e_f,e_q)
    up_gr_i=f_d[0];uf_gr_i=f_d[1];uq_gr_i=f_d[2]

    fac=mag_o/gyro_e
    e_p=fac*gr_p;e_f=fac*gr_f;e_q=fac*gr_q;
    f_d=drift(e_p,e_f,e_q)
    up_gr_e=f_d[3];uf_gr_e=f_d[4];uq_gr_e=f_d[5]
    
    cur_gr_p=np_t*q_charge*(up_gr_i-up_gr_e)
    cur_gr_f=np_t*q_charge*(uf_gr_i-uf_gr_e)
    cur_gr_q=np_t*q_charge*(uq_gr_i-uq_gr_e)   
    
#===AMBIENT ELECTRIC FIELD DRIVEN CURRENTS
    
    t_o=4000.;sigma_t=2000.;af=5.;bf=-2.
    ele_t=0.5#/(exp(af*(t_e/t_o-1.))+exp(bf*(t_e/t_o-1.))) #0.75
    ele_amb=-1.5e-03*ele_t
    
    ele_amb_p=i_amb*0;ele_amb_f=i_amb*ele_amb;ele_amb_q=i_amb*ele_amb
    f_d=drift(ele_amb_p,ele_amb_f,ele_amb_q)
    up_amb=f_d[3];uf_amb=f_d[4];uq_amb=f_d[5];

    cur_ele_p=s_11*ele_amb_p+s_12*ele_amb_f+s_13*ele_amb_q
    cur_ele_f=s_22*ele_amb_f+s_21*ele_amb_p+s_23*ele_amb_q
    cur_ele_q=s_33*ele_amb_q+s_31*ele_amb_p+s_32*ele_amb_f
    
#===TOTAL AMBIENT CURRENT/POTENTIAL SOURCE 
    
    cur_amb_p=cur_gr_p+cur_ele_p
    cur_amb_f=cur_gr_f+cur_ele_f
    cur_amb_q=cur_gr_q+cur_ele_q
            
    div_cur_amb=0*gradient(cur_amb_p)[0]/dp_i+gradient(cur_amb_f)[1]/df_i+0*gradient(cur_amb_q)[2]/dq_i
    source_amb=-1.*div_cur_amb/s_11 #2.5
#    fac=1./(sigma_p*np_t)
#    cur_amb_f=sigma_p*ele_amb_f+np_t*q_charge*gr_p/gyro_i
#    source_amb=-fac*cur_amb_f*gradient(np_t)[1]/df_i
    
#===>DIVERGENCE FREE CURRENT EQUATION
    
    df_map2=(s_11/s_22)*df_i**2.
    dq_map2=(s_11/s_33)*dq_i**2.
    
    a1=gradient(s_11)[0]/dp_i+gradient(s_21)[1]/df_i+i_o*gradient(s_31)[2]/dq_i
    a1=a1/s_11
    a2=gradient(s_12)[0]/dp_i+gradient(s_22)[1]/df_i+i_o*gradient(s_32)[2]/dq_i
    a2=a2/s_11
    a3=gradient(s_13)[0]/dp_i+gradient(s_23)[1]/df_i+i_o*gradient(s_33)[2]/dq_i
    a3=a3/s_11
   
    pot_g=pot
    w_iter=1.7
    for i_sor in range (0,n_sor):   
        #===
        gr_pot=gradient(pot_g)
        grad_pot_p=gr_pot[0]/dp_i;
        grad_pot_f=gr_pot[1]/df_i;
        grad_pot_q=gr_pot[2]/dq_i
        
        pot_1=a1*grad_pot_p;
        pot_2=a2*grad_pot_f;
        pot_3=a3*grad_pot_q
        
        gr_pot_pf=gradient(grad_pot_p)[1]/df_i
        gr_pot_pq=gradient(grad_pot_p)[2]/dq_i
        gr_pot_fp=gradient(grad_pot_f)[0]/dp_i
        gr_pot_fq=gradient(grad_pot_f)[2]/dq_i
        gr_pot_qp=gradient(grad_pot_q)[0]/dp_i
        gr_pot_qf=gradient(grad_pot_q)[1]/df_i
        
        pot_cross_1=s_12*gr_pot_fp+s_21*gr_pot_pf
        pot_cross_3=s_13*gr_pot_qp+s_31*gr_pot_pq
        pot_cross_2=s_23*gr_pot_qf+s_32*gr_pot_fq
        pot_cross=(pot_cross_1+pot_cross_2+pot_cross_3)/s_11
        
        a_p=(gradient(gr_pot[0])[0]+2.*pot_g)/dp_i**2.
        a_f=(gradient(gr_pot[1])[1]+2.*pot_g)/df_map2
        a_q=(gradient(gr_pot[2])[2]+2.*pot_g)/dq_map2
                        
        a=2.*(1./dp_i**2.+1./df_map2+i_o*1./dq_map2)
        
        pot=i_pot*(1./a)*(a_p+a_f+i_o*a_q+pot_1+pot_2+i_o*pot_3+i_o*pot_cross+source_wind+source_amb)
        
        delta_sor=abs(pot_g-pot).max()/pot.max()        
        pot_g=w_iter*pot+(1.-w_iter)*pot_g#(pot_g+pot)/2.        
        if delta_sor <1.e-02:
            print ('CONVERGENCE OF LAPLACE',i_sor)
            break
        
        w_iter=w_iter-0.1
        if w_iter<=1:
            w_iter=1.
                
#===POLARIZATION ELECTRIC FIELD
        
    ele_p=-gradient(pot)[0]/dp_i;
    ele_f=-gradient(pot)[1]/df_i;
    ele_q=-i_o*gradient(pot)[2]/dq_i
    
#===TOTAL ELECTRIC FIELD
    
    ele_tot_p=ele_amb_p+ele_p;
    ele_tot_f=ele_amb_f+ele_f;
    ele_tot_q=ele_amb_q+ele_q;

#    if up_tot.max()<5.e+02:
#        ele_tot_p=ele_p;ele_tot_f=ele_f;ele_tot_q=ele_q;

#===TOTAL ELECTRON DIRFT    

    f_d=drift(ele_tot_p,ele_tot_f,ele_tot_q)
    up_ele=f_d[3];uf_ele=f_d[4];uq_ele=f_d[5];

    f_d=drift(ele_tot_p,0*ele_f,0*ele_tot_q)
    uf_ele=f_d[4];

#    up_ele=-ele_tot_f/mag_o
#    uf_ele=ele_p/mag_o
    
#===>TOTAL DRIFT+WIND DRIFT
    
    up_tot=up_ele;uf_tot=uf_ele;uq_tot=uq_ele;    
    up_tot=up_ele+up_agw_i;uf_tot=uf_ele+uf_agw_i;uq_tot=uq_ele+uq_agw_i

    print ('AMBIENT, U_amb',round(up_amb.max(),2))
    print ('POTENTIAL, U_MAX', round(up_tot.max(),2),round(uf_tot.max(),2),round(uq_tot.max(),2))

    cur_tot_p=cur_wind_p+cur_amb_p
    cur_tot_f=cur_wind_f+cur_amb_f
    cur_tot_q=cur_wind_q+cur_amb_q
    
    gr_rate_incp=cur_tot_p*gradient(log(np_t))[0]/dp_i+cur_tot_f*gradient(log(np_t))[1]/df_i\
    /(np_t*q_charge)#-1.e-13*np_t
    
    gr_rate_cp=gradient(cur_tot_p)[0]/dp_i+gradient(cur_tot_f)[1]/df_i\
    /(np_t*q_charge)
    
#    print "GROWTH RATE",gr_rate_incp.max(),gr_rate_cp.max()
    return (pot,up_ele,up_amb,gr_rate_incp,gr_rate_cp,cur_tot_p,cur_tot_f)

#=============================================================================#
#==================================MAIN=======================================#
