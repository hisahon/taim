# from pyiri2016 import *#IRI2016,IRI2016Profile
# from iri2016 import *#IRI2016,IRI2016Profile
import iri2016.altitude as iri16
from nrlmsise_2000 import *
# from pymsis import *
from numpy import arange, isnan
from scipy.ndimage import *
from matplotlib.pyplot import figure, legend, show, subplots
from SIM_VOL import *
import datetime 

from comv import dates as devnt

import os
fname = os.path.basename(__file__)
print(fname)

global q_charge,b_c,mag_o
q_charge=1.6e-19;b_c=1.38e-23;eps_o=8.854e-12;m_i=1.67e-27;z_i=16.
mag_o=0.25e-04;

m=20.;nq_m=int(nq/m);nf_m=int(nf/m)


# np_i is que quantity of data thats exists higher than 80 km.
ne_iri_2d=empty((np_i,nf_m,nq_m))

def amb_atmos():
    global rho_msis,mean_mass,n_msis,tn_msis,pn_msis,sn_msis,nu_msis
    global r_go,tau_a,tau_b
    
    f=nrl_msis(doy,ut_sec,lt,f107A,f107,ap,lats,lons,altstp,alt_o,nalt)
    tn_msis=f[1];
    den_ox=f[2]*1.e+06;den_n=f[3]*1.e+06;den_o2=f[4]*1.e+06;den_n2=f[5]*1.e+06;
    n_msis=den_ox+den_n+den_o2+den_n2;
    rho_msis=f[6]*1.e+03 #kg/m-3 * 1000
    mean_mass=rho_msis/n_msis
    k_b=1.38e-23;r_go=k_b/mean_mass
    pn_msis=k_b*n_msis*tn_msis #Pa
    sn_msis=sqrt(1.4*pn_msis/rho_msis) #m/s
    nu_msis=pi*(7*5.6e-11)**2.*sn_msis*n_msis    
    visc_mu_1=3.563e-07*tn_msis**(0.71);
    visc_mu_2=1.3*pn_msis/nu_msis;
    
    gr_rho=gradient(rho_msis)/dp_m.max()
    gr_pn=gradient(pn_msis)/dp_m.max()
    gr_2_pn=gradient(gr_pn)/dp_m.max()        
    w_a=sqrt(1.4*pn_msis/rho_msis)*abs(gr_rho)/(2.*rho_msis);
    w_b=sqrt(abs(gr_pn*gr_rho/rho_msis**2.-gr_2_pn/rho_msis))
    tau_a=1./w_a;tau_b=1./w_b
    return ()

#===
   # def amb_iono():
   #     global ne_iri,no2p,ti,altbins
   #     altlim=[alt_i[0],alt_i[-1]]
   #     iri2016=IRI2016Profile(altlim=altlim,altstp=dr,lat=lats,\
   #         lon=lons, year=year, month=month, dom=dom, iut=iut,hour=hour,\
   #         option=1, verbose=False) 
       
       
   #     index = range(np_i)
       
   #     i_ne,i_no,i_no2,i_nno=0,4,7,8
   #     ne_iri = iri2016.a[0, index] # 
   #     nop = iri2016.a[4, index]*50.e+08# 
   #     no2p = iri2016.a[7, index]*10.e+08 # 
   #     nnop = iri2016.a[8, index]*10.e+08 #
           
   #     ti = iri2016.a[2, index]
   #     te = iri2016.a[3, index]
   #     return ()    



def amb_iono():
    global ne_iri,nop, no2p,nnop,ti,altbins, he_iri
    altlim=[alt_i[0],alt_i[-1],dr]
    # iri2016=IRI2016Profile(altlim=altlim,altstp=dr,lat=lats,\
    #     lon=lons, year=year, month=month, dom=dom, iut=iut,hour=hour,\
    #     option=1, verbose=False) 
    
    event_date = f'{year:04d}-{month:02d}-{dom:02d}T{int(hour):02d}:{int((hour - int(hour))*60):02d}'
        
    # # Replace the arguments with your specific values
    # time = "2003-11-21T12"
    # alt_km = [0,100,2]
    # glon = -76.77
    # glat = -11.95
    # # Run the altitude module
    f_iono=iri16.main(event_date, altlim, lats, lons)
    
    # index = range(np_i)
    
    # ["ne", "Tn", "Ti", "Te", "nO+", "nH+", "nHe+", "nO2+", "nNO+", "nCI", "nN+"]
    # i_ne,i_no,i_no2,i_nno=0,4,7,8
    
    ne_iri=f_iono['ne'].values
    # he_iri = (f_iono['nN+'].values)*5e9
    nop=(f_iono['nO+'].values)*5e9
    no2p=(f_iono['nO2+'].values)*1e9
    nnop=(f_iono['nNO+'].values)*1e9
    
    ti=f_iono['Ti'].values
    te=f_iono['Te'].values
    
    # ne_iri = iri2016.a[0, index] # 
    # nop = iri2016.a[4, index]*50.e+08# 
    # no2p = iri2016.a[7, index]*10.e+08 # 
    # nnop = iri2016.a[8, index]*10.e+08 #
        
    # ti = iri2016.a[2, index]
    # te = iri2016.a[3, index]
    return ()    

def gyro_coll():
    global omg_i,omg_e,nu_in,nu_en
    
    omg_i=q_charge*mag_o/(mean_mass[np-np_i:]);
    omg_e=-omg_i*1837.
    a=mean_mass[np-np_i:]/m_i #16.;#a is neutral mass in amu

    nu_ii=0.22*ne_iri*1.e-06/(ti)**(3./2.);
    nu_in=2.6e-9*1.e-06*(5*n_msis[np-np_i:]+ne_iri)*a**(-1./2.)        
    a=2.*sn_msis[np-np_i:]*8./(3.*sqrt(pi))
    nu_in=1.e+06*a*rho_msis[np-np_i:]
    nu_in=nu_in+0*nu_ii
    
    nu_ei=34*ne_iri*1.e-06/(ti)**(3./2.);
    nu_en=5.4e-16*n_msis[np-np_i:]*sqrt(ti)+0*nu_ei
    nu_en=nu_ei+0*nu_en
    
    kappa_i=omg_i/nu_in;kappa_e=omg_e/nu_en
    
    return ()


#%% running variables
i_mes=0;i_dia=0

alt_o, alt_f, altstp=alt[0],alt[-1],dr
altbins = arange(alt_o, alt_f+ altstp, altstp)
nalt = len(altbins)
                
                
lats, lons, ut= 3.29, 95.98, 02.5       
year, month, dom = 2004, 12+i_mes, 26+i_dia
# year, month, dom = 2021, 10+i_mes, 23+i_dia
# lats, lons, ut= -25, -58, 16
# year, month, dom, doy = devnt[0], devnt[1], devnt[2], devnt[3]


f107A,f107,ap=150, 150, 4

#        ut=lt-lons/15.
ut_sec=ut*3600.

#ut=13.
#ut_sec=ut*3600.
lt=ut+lons/15.

d0 = datetime.date(year, 1, 1)
d1 = datetime.date(year, month, dom)
delta = d1 - d0
doy=delta.days 

amb_atmos()


iut=1   #1 for UT, 0 for LT
hour=lt-iut*lons/15.
amb_iono()
gyro_coll()

if __name__ == '__main__':

    fig, axs = subplots(2,2,figsize=(12,12))
    ax = axs [0,0]
    ax.semilogx(rho_msis,alt,label=r'$\rho_o$')
    ax.semilogx(pn_msis,alt,label=r'pn $\rho_o$')
    # ax.set_xlim(1.e-12, 1.e+01)
    ax.set_ylim(0, 400)
    ax.set_xlabel('$kg/m^3$')
    ax.set_ylabel('Height, km')
    ax.set_title('a.', loc='left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    ax = axs [0,1]
    ax.semilogx(zoom(tn_msis[::3],3),zoom(alt[::3],3),label=r'$Temperature,$ K')
#    ax.semilogx(zoom(tau_a[::3],3)/60.,zoom(alt[::3],3),label=r'$\tau_a$, Minutes')
    # ax = ax.twiny()
    ax.semilogx(zoom(tau_b[::3],3)/60.,zoom(alt[::3],3),label=r'BV Period, min')
    ax.semilogx(zoom(tau_a[::3],3)/60.,zoom(alt[::3],3),label=r'Sound Period, min')
    ax.semilogx(zoom(sn_msis[::3],3)/60.,zoom(alt[::3],3),label=r'Sound Velocity, min')
    
    # ax.set_xlim(1e-1, 1500)
    ax.set_ylim(0, 400)
    # ax.set_xlabel('Time, min')
    ax.set_ylabel('Height, km')
    ax.set_title('b.', loc='left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()  
    
#==========IRI =========================

    no2p[no2p==0]=nan
    ax = axs [1,0]
    ax.semilogx(ne_iri, alt_i, label='$e^{{-}}$') 
    ax.semilogx(no2p, alt_i, label='$O_{{2}}^{{+}}$')
    

    # axis((1.e+09,1.e+12,0,400))
    # ax.set_xlim(1.e+09, 1.e+12)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Density (m$^{{-3}}$)')
    ax.set_ylabel('Height (km)')
    ax.set_title('c.', loc='left')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax = axs[1,1]
    kappa=omg_i/nu_in
    mu_p_i=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_i=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_i=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    kappa=omg_e/nu_en
    mu_p_e=(kappa/(mag_o*(1+kappa**2.)))
    mu_h_e=(kappa**2./(mag_o*(1+kappa**2.)))
    mu_o_e=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    sigma_p=q_charge*ne_iri*(mu_p_i-mu_p_e)
    sigma_h=-q_charge*ne_iri*(mu_h_i-mu_h_e)
    sigma_o=q_charge*ne_iri*(mu_o_i-mu_o_e)
    
    # ax=subplot(224)
    ax.semilogx(sigma_p,alt_i,label='$\sigma_P$')
    ax.semilogx(sigma_h,alt_i,label='$\sigma_H$')
    ax.semilogx(sigma_o,alt_i,label='$\sigma_{||}$')
    ax.set_xlim(1.e-09, 1.e+02)
    ax.set_ylim(0, 400)
    ax.set_xlabel('rho/m')
    ax.set_ylabel('Height, km')
    ax.set_title('d.', loc='left')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    show()
    
    #%% Secound Figure
    fig, axs = subplots(1,2,figsize=(10,5))
    ax = axs [0]
    ax.semilogx(rho_msis,alt,label=r'$\rho_o$, kg/m${^3}$')
    ax.semilogx(pn_msis,alt,label=r'pn, Pa')
    # ax.set_xlim(1.e-12, 1.e+01)
    ax.set_ylim(0, 400)
    # ax.set_xlabel('$kg/m^3$')
    ax.set_ylabel('Height, km')
    ax.set_title('a.', loc='left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    ax = axs [1]
    ax.semilogx(zoom(tn_msis[::3],3),zoom(alt[::3],3),label=r'Temperature, K')
#    ax.semilogx(zoom(tau_a[::3],3)/60.,zoom(alt[::3],3),label=r'$\tau_a$, Minutes')
    # ax = ax.twiny()
    ax.semilogx(zoom(tau_b[::3],3),zoom(alt[::3],3),label=r'BV Period, min')
    # ax.semilogx(zoom(tau_a[::3],3),zoom(alt[::3],3),label=r'Sound Period, min')
    ax.semilogx(zoom(sn_msis[::3],3),zoom(alt[::3],3),label=r'Sound Velocity, m/s')
    
    # ax.set_xlim(1e-1, 1500)
    ax.set_ylim(0, 400)
    # ax.set_xlabel('Time, min')
    ax.set_ylabel('Height, km')
    ax.set_title('b.', loc='left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()  
    
#==========IRI =========================
    
    fig, axs = subplots(1,2,figsize=(10,5)) 
    
    ax = axs [1]
    # ax.semilogx(rho_msis,alt,label=r'$\rho_o$, kg/m${^3}$')
    no2p[isnan(no2p)] = 1e-11
    # ax.semilogx(nop,alt_i,label=r'O$^+$')
    # ax.semilogx(no2p,alt_i,label=r'O$_2^+$')
    # ax.semilogx(ne_iri,alt_i,label=r'e$^-$')
    # ax.semilogx(he_iri, alt_i, label='H+')
    ax.semilogx(ne_iri+nop+no2p,alt_i,label=r'density')
    # ax.set_xlim(1.e-12, 1.e+01)
    ax.set_ylim(0, 400)
    # ax.set_xlabel('$kg/m^3$')
    ax.set_ylabel('Height, km')
    ax.set_xlabel('# of particles')
    ax.set_title('a.', loc='left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    ax = axs [0]
    ax.semilogx(zoom(tn_msis[::3],3),zoom(alt[::3],3),label=r'Temperature, K')
    
    
#    ax.semilogx(zoom(tau_a[::3],3)/60.,zoom(alt[::3],3),label=r'$\tau_a$, Minutes')
    # ax = ax.twiny()
    # ax.semilogx(zoom(tau_b[::3],3),zoom(alt[::3],3),label=r'BV Period, min')
    # ax.semilogx(zoom(tau_a[::3],3),zoom(alt[::3],3),label=r'Sound Period, min')
    # ax.semilogx(zoom(sn_msis[::3],3),zoom(alt[::3],3),label=r'Sound Velocity, m/s')
    
    # ax.set_xlim(1e-1, 1500)
    ax.set_ylim(0, 400)
    # ax.set_xlabel('Time, min')
    ax.set_ylabel('Height, km')
    ax.set_title('b.', loc='left')
    ax.set_xlabel('Temperature, K')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()  
    
    
    
    # no2p[no2p==0]=nan
    # ax = axs [1,0]
    # ax.semilogx(ne_iri, alt_i, label='$e^{{-}}$') 
    # ax.semilogx(no2p, alt_i, label='$O_{{2}}^{{+}}$')
    

    # # axis((1.e+09,1.e+12,0,400))
    # # ax.set_xlim(1.e+09, 1.e+12)
    # ax.set_ylim(0, 400)
    # ax.set_xlabel('Density (m$^{{-3}}$)')
    # ax.set_ylabel('Height (km)')
    # ax.set_title('c.', loc='left')
    # ax.legend(loc='upper left')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # ax = axs[1,1]
    # kappa=omg_i/nu_in
    # mu_p_i=(kappa/(mag_o*(1+kappa**2.)))
    # mu_h_i=(kappa**2./(mag_o*(1+kappa**2.)))
    # mu_o_i=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    # kappa=omg_e/nu_en
    # mu_p_e=(kappa/(mag_o*(1+kappa**2.)))
    # mu_h_e=(kappa**2./(mag_o*(1+kappa**2.)))
    # mu_o_e=(kappa*(1+kappa**2.)/(mag_o*(1+kappa**2.)))
    
    # sigma_p=q_charge*ne_iri*(mu_p_i-mu_p_e)
    # sigma_h=-q_charge*ne_iri*(mu_h_i-mu_h_e)
    # sigma_o=q_charge*ne_iri*(mu_o_i-mu_o_e)
    
    # # ax=subplot(224)
    # ax.semilogx(sigma_p,alt_i,label='$\sigma_P$')
    # ax.semilogx(sigma_h,alt_i,label='$\sigma_H$')
    # ax.semilogx(sigma_o,alt_i,label='$\sigma_{||}$')
    # ax.set_xlim(1.e-09, 1.e+02)
    # ax.set_ylim(0, 400)
    # ax.set_xlabel('rho/m')
    # ax.set_ylabel('Height, km')
    # ax.set_title('d.', loc='left')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    show()
    
    #Hisashi: comment done by me
    # ne_iri_2d[:,i_lon,i_lat]=ne_iri 
            
    #ne_iri_3d=interpolation.zoom(ne_iri_2d,(1,6,6),order=0)
    #save('ne_iri_3d.npy',ne_iri_3d)
    #imshow(ne_iri_2d.sum(0),origin='lower');axis('tight')
    #show()