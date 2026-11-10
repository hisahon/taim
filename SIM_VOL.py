import os
from pylab import *
from scipy import *
from numpy import sqrt, exp, ones
from numpy import gradient, linspace, zeros, arange, meshgrid, stack, hanning, outer, newaxis
from numpy import radians, sin, cos, tan, arctan,arccos, degrees
from comv import lat_i, lat_f, lon_i, lon_f, ltsb, lnsb, dphi, dtheta
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.io import netcdf
# from igrf import igrf
from ppigrf import igrf14_adapted
fname = os.path.basename(__file__)

def d1_3(n2,n3,data):
    return repeat(repeat(data[:,newaxis],n2,axis=1)[:,:,newaxis],n3,axis=2)

def inc_angle():
    # Declare global variable
    global angle_I
    
    # Calculate delta (angle between the magnetic field and the radial direction) and extract a subset
    delta = sqrt(1. + 3. * (sin(theta))**2.)
    delta_i = delta[np - np_i:, :, :]

    # Magnetic field components calculation
    b_o = 8.e+15  # Total magnetic field strength in Tesla * meter^3
    b_r = -2 * b_o * sin(theta) / r**3.  # Radial component of the magnetic field
    b_theta = b_o * cos(theta) / r**3.  # Theta (polar) component of the magnetic field
    b = b_o * delta / r**3.  # Total magnetic field strength

    # Calculate inclination angle with respect to the radial direction
    angle_inc = arccos(b_theta / b) #Bipolar
    angle_I = theta[np - np_i:, :, :]
    
    angle_I = angle_inc[np - np_i:, :, :]
    b_p = b_r / b  # Ratio of radial magnetic field component to total strength
    b_q = b_theta / b  # Ratio of theta component to total strength
    
    # Retrieve magnetic field variables for a specific date and location using igrf module
    f = igrf(event_date, glat=degrees(lat[0]), glon=degrees(lon[0]), alt_km=100).variables
    
    # Extract magnetic field components from the igrf module
    a = list(f.keys())
    [b_n, b_e, b_d, b_tot] = [f[a[0]][0], f[a[1]][0], f[a[2]][0], f[a[3]][0]]
    [angle_inc, angle_decl] = radians([f[a[5]][0], f[a[6]][0]])  # Convert inclination and declination to radians
    
    # Calculate inclination angle again using the igrf module
    inc_angle = arccos(b_n / b_tot)
    
    # print(angle_inc, inc_angle.values.item())




def hanning_edges_2d(data):
    # Obter as dimensões da matriz
    np, rows, cols = data.shape
    edge_size = 0.1
    edge_width_row = int(rows * edge_size)  # 15% das linhas para as bordas superior/inferior
    edge_width_col = int(cols * edge_size)  # 15% das colunas para as bordas esquerda/direita
    
    # Criar a janela de Hanning para as bordas superior/inferior e esquerda/direita
    hanning_row = hanning(2 * edge_width_row)[:edge_width_row]  # Metade da janela para bordas das linhas
    hanning_col = hanning(2 * edge_width_col)[:edge_width_col]  # Metade da janela para bordas das colunas
    
    # Máscara inicializada com 1s (não afetando o centro)
    mask = ones((rows, cols))

    # Aplicar o filtro de Hanning nas bordas superior e inferior
    mask[:edge_width_row, :] *= hanning_row[:, newaxis]  # Borda superior
    mask[-edge_width_row:, :] *= hanning_row[::-1][:, newaxis]  # Borda inferior

    # Aplicar o filtro de Hanning nas bordas esquerda e direita
    mask[:, :edge_width_col] *= hanning_col[newaxis, :]  # Borda esquerda
    mask[:, -edge_width_col:] *= hanning_col[::-1][newaxis, :]  # Borda direita

    # Replicar a máscara ao longo da primeira dimensão (np)
    mask_3d = stack([mask] * np, axis=0)

    return mask_3d

i_jump=1;n_sm=1
# event_date='2010-07-12'
event_date='2021-10-23'

r_ea=6371.
dt=10.
# dr=10.;
dr=10.; #Quando dr =20 perder frequencia alta
alt=arange(0,400. + dr, dr);np=len(alt)
i_alt=abs(alt-80).argmin()
alt_i=alt[i_alt:];np_i=len(alt_i)


## Original:
# dir_str='/home/alam/COLLABORADOR/IIG/MALA/ESTUDO_II/GRD'
# n_cd=len(dir_str)
# f= netcdf.netcdf_file(dir_str+'/surf.0.0.00000.grd','r')
# print (list(f.variables.keys()))
# a=list(f.variables.keys())
# [lon_0,lon_f]=f.variables[a[0]][:]
# [dphi,dtheta]=f.variables[a[3]][:]
# [lat_0,lat_f]=f.variables[a[1]][:]
# [nf,nq]=f.variables[a[4]][:]

## Altered Dec-2023

# [dphi,dtheta]=0.2, 0.2 #proportional to 0.04
# [lat_0,lat_f]=-35,-15
# [lon_0,lon_f]=-65,-45
[lon_0,lon_f]=lon_i,lon_f
[lat_0,lat_f]=lat_i,lat_f
# [nf,nq]=501,501

# lon_min, lon_max = -70., -40.
# lat_min, lat_max = -40., -10.

# lon_min, lon_max = -65., -45.
# lat_min, lat_max = -35., -15.

lon_min, lon_max = lon_i, lon_f
lat_min, lat_max = lat_i, lat_f


lon = arange(lon_0,lon_f+dtheta,dtheta)
lat = arange(lat_0,lat_f+dtheta,dtheta)



nf_min=abs(lon-lon_min).argmin()
nf_max=abs(lon-lon_max).argmin()

nq_min=abs(lat-lat_min).argmin()
nq_max=abs(lat-lat_max).argmin()

# nq=nq_max-nq_min;
# nf=nf_max-nf_min;


#%% Inclination angle
event_date_dt = datetime.strptime(event_date, "%Y-%m-%d")
B_igrf = igrf14_adapted(event_date_dt, lon[nf_min:nf_max], lat[nq_min:nq_max], alt_i, h=100)


lon=radians(lon[nf_min:nf_max])
lat=radians(lat[nq_min:nq_max])

#new values in radians
dphi=(abs(lon[1]-lon[0]));
dtheta=(abs(lat[1]-lat[0]));

nf=len(lon)
nq=len(lat)




iphi_m=abs(lon-lon.mean()).argmin()
iq_m=abs(lat-lat.mean()).argmin()
#iphi_m=iphi_m+int(iphi_m/4)
#iq_m=iq_m-int(iq_m/6)






mask_b = ones((np,nf,nq))


mask_b = hanning_edges_2d(mask_b)

smod = 4
mask_v = smod * (mask_b[0] * -1 + 1) + 1
mask_v = stack([mask_v] * np, axis=0)

# mask_v = ones((np,nf,nq))
#=============================================================================#

r=zeros((np,nf,nq));theta=zeros((np,nf,nq));phi=zeros((np,nf,nq))

for i in range (0,np):
    r[i,:,:]=(alt[i]+r_ea)*1.e+03;
for j in range (0,nf):
    phi[:,j,:]=lon[j]
for k in range (0,nq):
    theta[:,:,k]=lat[k]




# mask_b[-1,:,:]=0.0
# mask_b[-2,:,:]=0.2
# mask_b[-3,:,:]=0.4
# mask_b[-4,:,:]=0.8

# j_m=int((nf+1)/2);k_m=int((nq+1)/2);
# rad_m=sqrt(j_m**2.+k_m**2.)
        
        
# for k in range (0,nq):
#     for j in range (0,nf):
#         rad_o=sqrt((j-j_m)**2.+(k-k_m)**2.)
#         if rad_o >= 0.65*rad_m: # a partir de 60% de distancia horizontal, a amplitude diminui exponencialmente
#             mask_b[:,j,k]=exp(-(2.*rad_o/rad_m)**2.)

# from scipy.stats import multivariate_normal



#%% Variables Transformation
# Spherical Coordinates Transformation Expressions

# Transformed radial distance emphasizing polar angle:
r_o = r / (cos(theta))**2
# Represents the radial distance transformation, emphasizing the effect of the polar angle.

# Parameter delta related to the polar angle:
# delta = sqrt(1. + 0 * 3. * (sin(theta))**2.)
delta = 1.
# Here, delta is calculated but simplifies to sqrt(1), indicating minimal impact.

# Quantity q related to the transformed radial coordinate and polar angle:
q = r_o**3. * sin(theta) / r**2.
# Involves the cube of the transformed radial distance r_o, the sine of the polar angle, and the square of the original radial distance r.

# Another expression for the transformed radial coordinate:
p = r / cos(theta)**2.
# Similar to the first expression, representing the transformed radial distance.

# Quantity hq involving the ratio of r to r_o raised to the power of 3, divided by delta:
hq = (r / r_o)**3. / delta
# Represents a quantity derived from the ratio of the original and transformed radial distances, influenced by the polar angle.

# Component of radial distance in the azimuthal direction:
hphi = r * cos(theta)
# Represents the component of the radial distance in the azimuthal direction, influenced by the cosine of the polar angle.

# Quantity hp involving the cube of the cosine of theta divided by delta:
hp = cos(theta)**3. / delta
# Represents a quantity influenced by the cube of the cosine of the polar angle, divided by the parameter delta.

#%% Magnetic Field Transformation Expressions

# Gradient of p and q with respect to spatial coordinates:
grad_p = gradient(p)
grad_q = gradient(q)

# Magnitude of dp, dq, and df in magnetic coordinates:
dp = abs(grad_p[0] + abs(0 * grad_p[2]))
dq = abs(0 * grad_q[0] + grad_q[2])
# df = abs(dphi * hphi) #This guy is defined above

# Magnetic field components in magnetic coordinates:
dp_m = dp * hp
dq_m = dq * hq
df_m = dphi * hphi  # Corrected from df_m = df * dphi

# Magnetic field components in Cartesian coordinates:
x_f = hphi * tan(phi) * 1.e-03
y_f = p * hp * 1.e-03
z_f = q * hq * 1.e-03

# Magnetic field components in spherical coordinates:
x_sph = x_f
y_sph = y_f * delta
z_sph = z_f * delta

# Additional magnetic field components in Cartesian coordinates:
z_f = r * sin(theta) * sin(phi) * 1.e-03
x_f = r * sin(theta) * sin(phi) * 1.e-03
y_f = r * cos(theta) * 1.e-03

# Altitude, latitude, and longitude in magnetic coordinates:
alt_3 = r / 1.e+03 - r_ea
lat_3 = arctan(z_f / (y_f + 0.))
lon_3 = arctan(x_f / (y_f + 0.))


# dp=abs(gradient(p)[0]+abs(0*gradient(p)[2]))
# dq=abs(0*gradient(q)[0]+(gradient(q)[2]))

# dp_m=dp*hp;dq_m=dq*hq;df_m=dphi*hphi
# x_f=hphi*tan(phi)*1.e-03;y_f=p*hp*1.e-03;z_f=q*hq*1.e-03

# x_sph=x_f;y_sph=y_f*delta;z_sph=z_f*delta

# z_f=r*sin(theta)*sin(phi)*1.e-03;
# x_f=r*sin(theta)*sin(phi)*1.e-03
# y_f=r*cos(theta)*1.e-03

# alt_3=r/1.e+03-r_ea#y_sph-r_ea
# lat_3=(arctan(z_f/(y_f+0.)))
# lon_3=(arctan(x_f/(y_f+0.)))

for i in range (0,np):
    alt_3[i,:,:]=alt[i];
for k in range (0,nq):
    lat_3[:,:,k]=lat[k];
for j in range (0,nf):
    lon_3[:,j,:]=lon[j];
    
dp_m[:,:,:]=dp_m.max(); dq_m[:,:,:]=df_m.min(); df_m[:,:,:]=df_m.min();
#    dq_m=df_m
dp_i=dp_m[np-np_i:,:,:];df_i=df_m[np-np_i:,:,:];dq_i=dq_m[np-np_i:,:,:]




# mask_bound=1+zeros((np,nf,nq))







##rad_m=sqrt(iphi_m**2.+iq_m**2.)
##for k in range (0,nq):
##    mask_q=exp(-(k-iq_m)**2./iq_m**2.)
##    for j in range (0,int(nf)):
##        mask_phi=exp(-(k-iphi_m)**2./iphi_m**2.)
##        rad_o=sqrt(((n_m/nf)*(j-iphi_m))**2.+((n_m/nq)*(k-iq_m))**2.)
##        if rad_o >= 0.75*rad_m:
##            mask_bound[:,j,k]=exp(-(0.5*rad_o/rad_m)**2.)
#        if abs(k-iq_m)<0.8*iq_m and abs(j-iphi_m)<0.8*iphi_m:
#            mask_bound[:,j,k]=mask_q+mask_phi            

#=============================================================================#

# def inc_angle():
#     global angle_I
    
#     delta=sqrt(1.+3.*(sin(theta))**2.);delta_i=delta[np-np_i:,:,:]

#     b_o=8.e+15         #Tm^3
#     b_r=-2*b_o*sin(theta)/r**3.;b_theta=b_o*cos(theta)/r**3.
#     b=b_o*delta/r**3.
#     angle_inc=arccos(b_theta/b);angle_I=theta[np-np_i:,:,:]
#     b_p=b_r/b;b_q=b_theta/b
    
#     #Original
#     # import igrf12
#     # f=(igrf12.igrf('2010-07-12',glat=degrees(lat[0]),glon=degrees(lon[0]),alt_km=100)).variables
    
#     #Altered
#     import igrf
#     f=(igrf.igrf(event_date, glat=degrees(lat[0]), glon=degrees(lon[0]), alt_km=100)).variables

#     a=list(f.keys())
#     [b_n,b_e,b_d,b_tot]=[f[a[0]][0],f[a[1]][0],f[a[2]][0],f[a[3]][0]]
#     # [angle_inc,angle_decl]=radians([f[a[5]][0],f[a[6]][0]]) #Original
#     [angle_inc,angle_decl]=radians([f[a[5]][0],f[a[6]][0]])
    
#     inc_angle=arccos(b_n/b_tot)
    
#     print (angle_inc,inc_angle)

#     return()




    
# f=inc_angle()

#%% Main running
if __name__ == '__main__':
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(degrees(lon), degrees(lat), mask_b[0].T, cmap='viridis')
    # plt.axhline(y=-65, color='k', linestyle='--')
    # plt.axhline(y=-45, color='k', linestyle='--')
    # plt.axvline(x=-35, color='k', linestyle='--')
    # plt.axvline(x=-15, color='k', linestyle='--')
    plt.colorbar(label='Intensity')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Normalized Mask of the boundaries')
    plt.show()
    
    # Plot inclination Angle
    plt.figure(figsize=(8, 5))
    # plt.pcolormesh(rad2deg(lat), rad2deg(lon), angle_I[0], cmap='Reds_r')
    # plt.contourf(lat, lon, angle_I[0], cmap='Reds_r', levels = 5)
    plt.contourf(degrees(lon), degrees(lat), angle_I[0].T, cmap='Reds_r', levels = 5)
    # plt.axhline(y=-65, color='k', linestyle='--')
    # plt.axhline(y=-45, color='k', linestyle='--')
    # plt.axvline(x=-35, color='k', linestyle='--')
    # plt.axvline(x=-15, color='k', linestyle='--')
    plt.colorbar(label='Intensity')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Normalized Mask of the boundaries')
    plt.show()



    
    plt.show()

    #Grid Properties
    print ('=========GRID_SIZES============')
    print (f'file: {fname}')
    print (f'hight: {np}, lat {nq}, lo {nf}')
    print (f'lon: {lon_0}, {lon_f}, {round(dphi,4)} rad')
    print (f'lat: {lat_0}, {lat_f}, {round(dtheta,4)} rad')
    print (f'dp_min | dp_max: {round(dp_m.min())} | {round(dp_m.max())}')
    print (f'dq_min | dq_max: {round(dq_m.min())} | {round(dq_m.max())}')
    print (f'df_min | df_max: {round(df_m.min())} | {round(df_m.max())}')
    print ('===============================')
#=============================================================================#

from AMBIENT import *

##ne=load('ne_iri_3d.npy')
##ne_iri_3d=ne[:,:,2:];
##ne_iri_3d[:,:,0:2]=ne_iri_3d[:,:,2:4]
##np_amb=ne_iri_3d;

rho_o=d1_3(nf,nq,rho_msis)
n_n=d1_3(nf,nq,n_msis)
tn_o=d1_3(nf,nq,tn_msis)
r_g=d1_3(nf,nq,r_go)
nu_nn=d1_3(nf,nq,nu_msis)
np_amb=d1_3(nf,nq,ne_iri)
nu_i=d1_3(nf,nq,nu_in)
nu_e=d1_3(nf,nq,nu_en)
gyro_i=d1_3(nf,nq,omg_i)
gyro_e=d1_3(nf,nq,omg_e)
