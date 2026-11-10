# -*- coding: utf-8 -*-
# import numpy as np

from numpy import gradient, exp, sqrt, power, pi, log, zeros_like

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# def setup_ax(ax, extd):
#     ax.coastlines(resolution='110m', color='k')
#     ax.add_feature(cfeature.BORDERS)
#     ax.set_extent(extd, crs=ccrs.PlateCarree())
    
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
    
#     gl.xlocator = MaxNLocator(integer=True)
#     gl.ylocator = MaxNLocator(integer=True)


def data_antes(ndim, ndata, data):
    # Copy data and shift values to the left
    data_n = zeros_like(data)
    if ndim == 1:
        data_n[1:-1,:,:] = data[0:-2,:,:]  #Copy the [0 to n-2] column into [1 to n-1]
        data_n[0,:,:] = data_n[1,:,:]  # Copy the second column to the first column
        data_n[-1,:,:] = data_n[-2,:,:]  # Copy the second-to-last column to the last column
    return data_n

#=============================================================================#
def ndf(x,mu,sig):
    return exp(-power((x-mu)/sig, 2.)/2)
#=============================================================================#
def div_f(f_p,f_q,f_f,dp,dq,df):
    
    div= gradient(f_p, axis=0, edge_order=2)/dp\
        +gradient(f_q, axis=1, edge_order=2)/dq\
        +gradient(f_f, axis=2, edge_order=2)/df
        
    return (div)        
#=============================================================================#
def modes(gamma,rho,tn,pn,dp): #Utilizes dp_m because i changed
    grad_rho=gradient(rho)[0]/dp
    grad_pn=gradient(pn)[0]/dp
    grad_2_pn=gradient(grad_pn)[0]/dp        
    w_a=sqrt(gamma*pn/rho)*abs(grad_rho)/(2.*rho);
    w_b=sqrt(abs(grad_pn*grad_rho/rho**2.-grad_2_pn/rho))
    tau_a, tau_b = 2.*pi/w_a, 2.*pi/w_b
    
    return (tau_a,tau_b)
#=============================================================================#
# def dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_o,rho_amb,tn_o, nu_nn,dp,dq,df,dt):
def dissip_coef2(mask_bound,wp_o,wq_o,wf_o, amb_v,dts,coefs):
    
    pn_amb, rho_amb, rho_o, _, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    vcoef, hcoef, _ = coefs[0], coefs[1],  coefs[2]
    dp, dq, df, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
    
    # visc_mu = 3.563e-07*tn_o**(0.71)
    # visc_ki = visc_mu/rho_o
    
    #Calculationg w_visc_rho----------------------------------------------------
    visc_mu=1.3*pn_amb/nu_nn
    
    gr_flux= wp_o* gradient(rho_amb, axis=0)/dp
    w_visc_p=(1./dp**2.)*gr_flux*visc_mu/rho_amb**2.    
    
    gr_flux=wf_o* gradient(rho_amb, axis=1)/df
    w_visc_f=(1./df**2.)*gr_flux*visc_mu/rho_amb**2.
    
    gr_flux=wq_o * gradient(rho_amb, axis=2)/dq
    w_visc_q=(1./dq**2.)*gr_flux*visc_mu/rho_amb**2.
            
    w_visc_rho=abs(w_visc_p)+abs(w_visc_f)+abs(w_visc_q)
    
    #Calculationg w_visc_w------------------------------------------------------
    visc_ki=visc_mu/rho_amb
    w_visc_w=(1./dp**2. + 1./df**2. + 1./dq**2.)*visc_ki/dt
     
    #Calculationg w_visc--------------------------------------------------------
    w_visc=exp(-0.5*dt**2.*(abs(w_visc_rho)+abs(w_visc_w)))
    
    #=====================================================================#

    fac_nl=wp_o.max()**2./(dp*dt)
    w_nl=exp(-1*vcoef*dt**2.*fac_nl)
    wp_damp=w_nl*w_visc  
    
    fac_nl=wf_o.max()/(df*dt) + wq_o.max()/(dq*dt)
    w_nl=exp(-1*hcoef*dt**2.*fac_nl)
    w_damp=w_nl*w_visc 
    
    # print(dp[0,0,0], df[0,0,0], dq[0,0,0])
    
    # fac_nl=wp_o.max()/dp + wf_o.max()/df + wq_o.max()/dq
    # w_nl=exp(-0.05*dt**2.*fac_nl/dt)
    # w_damp=w_nl*w_visc*mask_bound
    
    return (wp_damp,w_damp)

def dissip_coef(mask_bound,wp_o,wq_o,wf_o,amb_v,dp,dq,df,dt,coefs):
    
    pn_amb, rho_amb, rho_o, _, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    vcoef, hcoef, _ = coefs[0], coefs[1],  coefs[2]
    # dp, dq, df, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
    
    # visc_mu = 3.563e-07*tn_o**(0.71)
    # visc_ki = visc_mu/rho_o
    
    #Calculationg w_visc_rho----------------------------------------------------
    visc_mu=1.3*pn_amb/nu_nn
    
    gr_flux= wp_o* gradient(rho_amb, axis=0)/dp
    w_visc_p=(1./dp**2.)*gr_flux*visc_mu/rho_amb**2.    
    
    gr_flux=wf_o* gradient(rho_amb, axis=1)/df
    w_visc_f=(1./df**2.)*gr_flux*visc_mu/rho_amb**2.
    
    gr_flux=wq_o * gradient(rho_amb, axis=2)/dq
    w_visc_q=(1./dq**2.)*gr_flux*visc_mu/rho_amb**2.
            
    w_visc_rho=abs(w_visc_p)+abs(w_visc_f)+abs(w_visc_q)
    
    #Calculationg w_visc_w------------------------------------------------------
    visc_ki=visc_mu/rho_amb
    w_visc_w=(1./dp**2. + 1./df**2. + 1./dq**2.)*visc_ki/dt
     
    #Calculationg w_visc--------------------------------------------------------
    w_visc=exp(-0.5*dt**2.*(abs(w_visc_rho)+abs(w_visc_w)))
    
    #=====================================================================#

    fac_nl=wp_o.max()**2./(dp*dt)
    w_nl=exp(-1*vcoef*dt**2.*fac_nl)
    wp_damp=w_nl*w_visc  
    
    fac_nl=wf_o.max()/(df*dt) + wq_o.max()/(dq*dt)
    w_nl=exp(-1*hcoef*dt**2.*fac_nl)
    w_damp=w_nl*w_visc 
    
    # print(dp[0,0,0], df[0,0,0], dq[0,0,0])
    
    # fac_nl=wp_o.max()/dp + wf_o.max()/df + wq_o.max()/dq
    # w_nl=exp(-0.05*dt**2.*fac_nl/dt)
    # w_damp=w_nl*w_visc*mask_bound
    
    return (wp_damp,w_damp)
#==========================================================================#
# # def AGW_2(iax,gamma,delta,pn,rho,div_w,div_flux,w_gr_pn):
def AGW_DAMP(iax,gamma,delta,pn,rho,div_w,div_flux,w_gr_pn):
    
    flux_p=gamma*pn*div_w
    grad=gradient(flux_p)
    grad_flux=grad[iax]/delta
    w_1=grad_flux/rho 
    
    grad=gradient(pn);grad_pn=abs(grad[iax])/delta    #IMPORTANT OF USING ABS
    rho_m=(gradient(gradient(rho)[0])[0]+4.*rho)/(4.) #Nessa parte o programa usa a variação de rho da altitude mas na verdade eu deveria pegar rho para cara eixo
    w_2=-grad_pn*div_flux/rho_m**2.    
    
    grad=gradient(w_gr_pn)
    w_3=grad[iax]/delta
    
    return (w_1,w_2,w_3)    

def AWI(gamma, pn, div_w, deltas, rho):
    flux_p = gamma * pn * div_w
    grad = gradient(flux_p, edge_order=2)
    
    # grad_flux = grad[iax] / delta
    # w_1 = grad_flux / rho
    
    grad_flux_rho = grad/rho
    AWIp = grad_flux_rho[0]/deltas[0]
    AWIf = grad_flux_rho[1]/deltas[1]
    AWIq = grad_flux_rho[2]/deltas[2]
    
    return AWIp, AWIf, AWIq


def AWII(gamma, pn, div_w, deltas, rho):
    # grad = gradient(pn)
    grad = gradient(pn, edge_order=2)/rho
    
    grad_pnp = abs(grad[0]) / deltas[0]
    grad_pnf = abs(grad[1]) / deltas[1]
    grad_pnq = abs(grad[2]) / deltas[2]

    
    grad_flux_rho = grad/rho
    AWIIp = (gamma - 1) * grad_pnp * div_w
    AWIIf = (gamma - 1) * grad_pnf * div_w
    AWIIq = (gamma - 1) * grad_pnq * div_w
    
    return AWIIp, AWIIf, AWIIq

def GWI(pn, div_flux, deltas, rho):
    
    grad = gradient(pn, edge_order=2)

    # rho_m = (gradient(gradient(rho)[0])[0] + 4. * rho) / 4.
    
    # #porque vcalores absolutos
    # grad_pnp = abs(grad[0]) / deltas[0]
    # grad_pnf = abs(grad[1]) / deltas[1]
    # grad_pnq = abs(grad[2]) / deltas[2]
    
    # GWIp = - grad_pnp * div_flux / rho_m**2.
    # GWIf = - grad_pnf * div_flux / rho_m**2.
    # GWIq = - grad_pnq * div_flux / rho_m**2.
    
    rho_m=data_antes(1, rho.shape[0], rho)
    div_flux_rho = div_flux / rho_m**2
    
    # div_flux_rho = div_flux / rho
    
    # grad_pnp = grad[0] / deltas[0]
    # grad_pnf = grad[1] / deltas[1]
    # grad_pnq = grad[2] / deltas[2]
    
    
    GWIp = - grad[0] * div_flux_rho / deltas[0]
    GWIf = - grad[1] * div_flux_rho / deltas[1]
    GWIq = - grad[2] * div_flux_rho / deltas[2]
    
    return GWIp, GWIf, GWIq

def GWII(w_gr_pn, deltas, rho):
    
    grad = gradient(w_gr_pn, edge_order=2) / rho
    GWIIp = grad[0] / deltas[0]
    GWIIf = grad[1] / deltas[1]
    GWIIq = grad[2] / deltas[2]
    
    return GWIIp, GWIIf, GWIIq

def RTgr(din, dt, rho, deltas):
    
    R = 8.31446261815324 # J * K^{-1} * mol^{-1}
    grad = gradient(log(rho))
    
    # dtn = din/dt #T/s
    dtn = din
    
    
    rtp = R * dtn * grad[0] / deltas[0]
    rtf = R * dtn * grad[1] / deltas[1]
    rtq = R * dtn * grad[2] / deltas[2]
    
    return rtp, rtf, rtq
    
#====
def w_nl(ao,am,wp_o,wq_o,wf_o,wp_m,wq_m,wf_m,dt):
    
    # b=wp_o*gradient(ao)[0]+wf_o*gradient(ao)[1]+wq_o*gradient(ao)[2]
    # c=wp_m*gradient(am)[0]+wf_m*gradient(am)[1]+wq_m*gradient(am)[2]
    # d=(b-c)/dt
    
    # Compute gradients along all axes for ao and am
    grad_ao = gradient(ao, edge_order=2)
    grad_am = gradient(am, edge_order=2)
    
    # Compute b and c using vectorized operations
    b = wp_o * grad_ao[0] + wf_o * grad_ao[1] + wq_o * grad_ao[2]
    c = wp_m * grad_am[0] + wf_m * grad_am[1] + wq_m * grad_am[2]

    # Compute d
    d = (b - c) / dt
    
    return d
#=============================================================================#

    # amb_v = [pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn]
    
    # f=agw.AGW_AMP(gamma, mask_bound, w_o, w_m, dts, dTin, amb_v, i_t, n_sor)

def AGW_AMP(gamma, mask_bound, w_o, w_m, dts, amb_v, coefs):
    # global wp,wf,wq
    
    # deltas = [dp_m,df_m,dq_m]
    
    pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    _, _, n_sor = coefs[0], coefs[1],  coefs[2]
    
    wp_m, wq_m, wf_m = w_m[0], w_m[1], w_m[2]
    wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
    dp_m, dq_m, df_m, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
    # dp_m, dq_m, df_m, dt = dts[0], dts[1]*2.5, dts[2]*2.5, dts[3]
    
    wpozero = wp_o[0,:,:]
    
    wp, wf, wq = wp_o, wf_o, wq_o
    rho,tn = rho_o,tn_o
    
    #+++++++++++++++++++++++++++
    pn=r_g*rho*tn
    c_s=sqrt(gamma*pn/rho);  
    
    lambda_c=c_s**2./nu_nn
    
    wp_0 = 2*wp_o - wp_m;
    wf_0 = 2*wf_o - wf_m;
    wq_0 = 2*wq_o - wq_m;
        
    for i_sor in range (0,n_sor):
        
        wp[-1, :, :] = 0
        
        #=====================================================================#
        
        # div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        # div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        gr_pn=gradient(pn, edge_order=2) #C
        w_gr_pn=wp*gr_pn[0]/dp_m + wf*gr_pn[1]/df_m + wq*gr_pn[2]/dq_m #D
    
        #=====================================================================#
        
        
        # kherani 2012 book chapter
        awip, awif, awiq = AWI(gamma, pn, div_w, dts, rho)
        # awiip, awiif, awiiq = AWII(gamma, pn, div_w, dts, rho) #this term does exists in kherani 2012
        gwip, gwif, gwiq = GWI(pn, div_flux, dts, rho)
        gwiip, gwiif, gwiiq = GWII(w_gr_pn, dts, rho)
        
        rtp, rtf, rtq = RTgr(dTin, dt, rho, dts)
        
        # f_p=AGW_DAMP(0,gamma,dp_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_f=AGW_DAMP(1,gamma,df_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_q=AGW_DAMP(2,gamma,dq_m,pn,rho,div_w,div_flux,w_gr_pn) 
        
        #=====================================================================#
        # wp_o[0,:,:] = wpozero
        # f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_amb,rho_o,tn_o, nu_nn,dp_m,dq_m,df_m,dt)
        f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, amb_v,dp_m,dq_m,df_m,dt, coefs)
        wp_damp=f[0];w_damp=f[1]
        
        # d2_w=    gradient(gradient(wp_o)[0])[0]/dp_m**2.\
        #         +gradient(gradient(wf_o)[1])[1]/df_m**2.\
        #         +gradient(gradient(wq_o)[2])[2]/dq_m**2.
        
        # c_visc=exp(-1.*abs((lambda_mu/rho)*d2_w*t/max(10,wp_o.max())))
        # c_diff=1.#ndf(t,0,10)#*c_visc 
        
        
        temp_term = [rtp, rtf, rtq]
        
        wp=wp_damp*(wp_0 + dt**2.*(awip + gwip + gwiip - 0*rtp))*mask_bound
        wf= w_damp*(wf_0 + dt**2.*(awif + gwif + gwiif - 0*rtf))*mask_bound
        wq= w_damp*(wq_0 + dt**2.*(awiq + gwiq + gwiiq - 0*rtq))*mask_bound
        
        #Here appears that the program solves the temperature and pressure 
        #=====================================================================#
    
    for i_sor in range (0,n_sor):
        
        div_flux_o=div_f(0*rho_o*wp_o,rho_o*wf_o,rho_o*wq_o,dp_m,df_m,dq_m) #non0
        div_flux  =div_f(0*rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #non 0
        
        # d2_rho= gradient(gradient(rho_o)[0])[0]/dp_m**2.\
        #        +gradient(gradient(rho_o)[1])[1]/df_m**2.\
        #        +gradient(gradient(rho_o)[2])[2]/dq_m**2.
        
        # c_d=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
        
        
        #just to see what happen Hisashi
        # rho=rho_o-0.5*dt*(div_flux+div_flux_o)/2.
        rho=rho_o - 0*dt*(div_flux+div_flux_o)/2.
        
        #================================================#
        
        div_flux_o=div_f(tn_o*wp_o,tn_o*wf_o,tn_o*wq_o, dp_m,df_m,dq_m)        
        div_flux=div_f(tn*wp,tn*wf,tn*wq, dp_m,df_m,dq_m)        
        div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)
        
        d2_tn= gradient(gradient(tn_o)[0])[0]/dp_m**2.\
              +gradient(gradient(tn_o)[1])[1]/df_m**2.\
              +gradient(gradient(tn_o)[2])[2]/dq_m**2.
        
        c_d=exp(-0.5*abs(lambda_c*d2_tn*dt/tn_o))
        
        tn=(tn_o-0*dt*c_d*((div_flux+div_flux_o)/2.\
                        +(gamma-1)*tn_o*div_w\
                        -0*lambda_c*d2_tn))
            
        # tn=(tn_o-0.5*dt*c_d*((div_flux+div_flux_o)/2.\
        #                 +(gamma-1)*tn_o*div_w\
        #                 -0*lambda_c*d2_tn))
    
    return (wp,wf,wq, rho,tn, temp_term)

def AGW_AMP2(w_o, w_m, dts, amb_v, coefs):
    # global wp,wf,wq
    
    # deltas = [dp_m,df_m,dq_m]
    
    pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    _, _, n_sor, gamma, mask_bound = coefs[0], coefs[1],  coefs[2], coefs[3], coefs[4]
    
    wp_m, wq_m, wf_m = w_m[0], w_m[1], w_m[2]
    wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
    dp_m, dq_m, df_m, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
    # dp_m, dq_m, df_m, dt = dts[0], dts[1]*2.5, dts[2]*2.5, dts[3]
    
    wpozero = wp_o[0,:,:]
    
    wp, wf, wq = wp_o, wf_o, wq_o
    rho,tn = rho_o,tn_o
    
    #+++++++++++++++++++++++++++
    pn=r_g*rho*tn
    c_s=sqrt(gamma*pn/rho);  
    
    lambda_c=c_s**2./nu_nn
    
    wp_0 = 2*wp_o - wp_m;
    wf_0 = 2*wf_o - wf_m;
    wq_0 = 2*wq_o - wq_m;
        
    for i_sor in range (0,n_sor):
        
        wp[-1, :, :] = 0
        
        #=====================================================================#
        
        # div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        # div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        gr_pn=gradient(pn, edge_order=2) #C
        w_gr_pn=wp*gr_pn[0]/dp_m + wf*gr_pn[1]/df_m + wq*gr_pn[2]/dq_m #D
    
        #=====================================================================#
        
        
        # kherani 2012 book chapter
        awip, awif, awiq = AWI(gamma, pn, div_w, dts, rho)
        # awiip, awiif, awiiq = AWII(gamma, pn, div_w, dts, rho) #this term does exists in kherani 2012
        gwip, gwif, gwiq = GWI(pn, div_flux, dts, rho)
        gwiip, gwiif, gwiiq = GWII(w_gr_pn, dts, rho)
        
        rtp, rtf, rtq = RTgr(dTin, dt, rho, dts)
        
        # f_p=AGW_DAMP(0,gamma,dp_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_f=AGW_DAMP(1,gamma,df_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_q=AGW_DAMP(2,gamma,dq_m,pn,rho,div_w,div_flux,w_gr_pn) 
        
        #=====================================================================#
        # wp_o[0,:,:] = wpozero
        # f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_amb,rho_o,tn_o, nu_nn,dp_m,dq_m,df_m,dt)
        f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, amb_v,dp_m,dq_m,df_m,dt, coefs)
        wp_damp=f[0];w_damp=f[1]
        
        # d2_w=    gradient(gradient(wp_o)[0])[0]/dp_m**2.\
        #         +gradient(gradient(wf_o)[1])[1]/df_m**2.\
        #         +gradient(gradient(wq_o)[2])[2]/dq_m**2.
        
        # c_visc=exp(-1.*abs((lambda_mu/rho)*d2_w*t/max(10,wp_o.max())))
        # c_diff=1.#ndf(t,0,10)#*c_visc 
        
        
        temp_term = [rtp, rtf, rtq] #Export this if you want to see it
        
        # wp_damp[:] = 1; w_damp[:] = 1
        wp=wp_damp*(wp_0 + dt**2.*(awip + gwip + gwiip - 0*rtp))*mask_bound
        wf= w_damp*(wf_0 + dt**2.*(awif + gwif + gwiif - 0*rtf))*mask_bound
        wq= w_damp*(wq_0 + dt**2.*(awiq + gwiq + gwiiq - 0*rtq))*mask_bound
        
        #Here appears that the program solves the temperature and pressure 
        #=====================================================================#
    
    # for i_sor in range (0,n_sor):
        
    #     div_flux_o=div_f(0*rho_o*wp_o,rho_o*wf_o,rho_o*wq_o,dp_m,df_m,dq_m) #non0
    #     div_flux  =div_f(0*rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #non 0
        
    #     # d2_rho= gradient(gradient(rho_o)[0])[0]/dp_m**2.\
    #     #        +gradient(gradient(rho_o)[1])[1]/df_m**2.\
    #     #        +gradient(gradient(rho_o)[2])[2]/dq_m**2.
        
    #     # c_d=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
        
        
    #     #just to see what happen Hisashi
    #     # rho=rho_o-0.5*dt*(div_flux+div_flux_o)/2.
    #     rho=rho_o - 0*dt*(div_flux+div_flux_o)/2.
        
    #     #================================================#
        
    #     div_flux_o=div_f(tn_o*wp_o,tn_o*wf_o,tn_o*wq_o, dp_m,df_m,dq_m)        
    #     div_flux=div_f(tn*wp,tn*wf,tn*wq, dp_m,df_m,dq_m)        
    #     div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)
        
    #     d2_tn= gradient(gradient(tn_o)[0])[0]/dp_m**2.\
    #           +gradient(gradient(tn_o)[1])[1]/df_m**2.\
    #           +gradient(gradient(tn_o)[2])[2]/dq_m**2.
        
    #     c_d=exp(-0.5*abs(lambda_c*d2_tn*dt/tn_o))
        
    #     tn=(tn_o-0*dt*c_d*((div_flux+div_flux_o)/2.\
    #                     +(gamma-1)*tn_o*div_w\
    #                     -0*lambda_c*d2_tn))
            
    #     # tn=(tn_o-0.5*dt*c_d*((div_flux+div_flux_o)/2.\
    #     #                 +(gamma-1)*tn_o*div_w\
    #     #                 -0*lambda_c*d2_tn))
    
    return (wp,wf,wq, rho,tn)

# from numba import njit

# @njit(parallel=True)
def AGW_AMP3(w_o, w_m, dts, amb_v, coefs):
    
    pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    _, _, n_sor, gamma, mask_bound = coefs[0], coefs[1],  coefs[2], coefs[3], coefs[4]
    
    wp_m, wq_m, wf_m = w_m[0], w_m[1], w_m[2]
    wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
    dp_m, dq_m, df_m, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
    # dp_m, dq_m, df_m, dt = dts[0], dts[1]*2.5, dts[2]*2.5, dts[3]
    
    # wpozero = wp_o[0,:,:]
    
    wp, wf, wq = wp_o, wf_o, wq_o
    rho, tn = rho_o, tn_o
    
    #+++++++++++++++++++++++++++
    pn=r_g*rho*tn
    c_s=sqrt(gamma*pn/rho);  
    
    lambda_c=c_s**2./nu_nn
    
    wp_0 = 2*wp_o - wp_m;
    wf_0 = 2*wf_o - wf_m;
    wq_0 = 2*wq_o - wq_m;
        
    # f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, amb_v,dp_m,dq_m,df_m,dt, coefs)
    for i_sor in range (0,n_sor):
        
        # wp[0, :, :] = 0
        
        #=====================================================================#
        
        # div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        # div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        div_w=div_f(wp, wf, wq, dp_m, df_m, dq_m)  #A     
        div_flux=div_f(rho*wp, rho*wf, rho*wq, dp_m, df_m, dq_m) #B
        
        gr_pn=gradient(pn, edge_order=2) #C
        w_gr_pn=wp*gr_pn[0]/dp_m + wf*gr_pn[1]/df_m + wq*gr_pn[2]/dq_m #D
    
        #=====================================================================#
        
        
        # kherani 2012 book chapter
        awip, awif, awiq = AWI(gamma, pn, div_w, dts, rho)
        # awiip, awiif, awiiq = AWII(gamma, pn, div_w, dts, rho) #this term does exists in kherani 2012
        gwip, gwif, gwiq = GWI(pn, div_flux, dts, rho)
        gwiip, gwiif, gwiiq = GWII(w_gr_pn, dts, rho)
        
        rtp, rtf, rtq = RTgr(dTin, dt, rho, dts)
        
        # f_p=AGW_DAMP(0,gamma,dp_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_f=AGW_DAMP(1,gamma,df_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_q=AGW_DAMP(2,gamma,dq_m,pn,rho,div_w,div_flux,w_gr_pn) 
        
        #=====================================================================#
        # wp_o[0,:,:] = wpozero
        # f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_amb,rho_o,tn_o, nu_nn,dp_m,dq_m,df_m,dt)
        # f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, amb_v,dp_m,dq_m,df_m,dt, coefs)
        f=dissip_coef2(mask_bound, wp, wq, wf, amb_v, dts, coefs)
        wp_damp=f[0];w_damp=f[1]
        
        # d2_w=    gradient(gradient(wp_o)[0])[0]/dp_m**2.\
        #         +gradient(gradient(wf_o)[1])[1]/df_m**2.\
        #         +gradient(gradient(wq_o)[2])[2]/dq_m**2.
        
        # c_visc=exp(-1.*abs((lambda_mu/rho)*d2_w*t/max(10,wp_o.max())))
        # c_diff=1.#ndf(t,0,10)#*c_visc 
        
        
        # temp_term = [rtp, rtf, rtq] #Export this if you want to see it
        
        # wp_damp[:] = 1 ; w_damp[:] = 1
        wp= wp_damp*(wp_0 + dt**2.*(awip + gwip + gwiip - 0*rtp))*mask_bound
        wf= w_damp*(wf_0 + dt**2.*(awif + gwif + gwiif - 0*rtf))*mask_bound
        wq= w_damp*(wq_0 + dt**2.*(awiq + gwiq + gwiiq - 0*rtq))*mask_bound
        
        # wp = wp_damp
        # wf = w_damp
        # wq = w_damp
    
    return (wp, wf, wq)


def AGW_DENSITY_TEMPERATURE(w_o, ws, dts, amb_v, coefs, operational=True):
    
    _, _, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    rho,tn = rho_o,tn_o
    
    if operational:
        _, _, n_sor, gamma, _ = coefs[0], coefs[1],  coefs[2], coefs[3], coefs[4]
        wp, wq, wf = ws[0], ws[1], ws[2]
        wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
        dp_m, dq_m, df_m, dt, _ = dts[0], dts[1], dts[2], dts[3], dts[4]
        
        pn=r_g*rho*tn
        c_s=sqrt(gamma*pn/rho)
        lambda_c=c_s**2./nu_nn
        
        for i_sor in range (0,n_sor):
            
            div_flux_o=div_f(0*rho_o*wp_o,rho_o*wf_o,rho_o*wq_o,dp_m,df_m,dq_m) #non0
            div_flux  =div_f(0*rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #non 0
            
            # d2_rho= gradient(gradient(rho_o)[0])[0]/dp_m**2.\
            #        +gradient(gradient(rho_o)[1])[1]/df_m**2.\
            #        +gradient(gradient(rho_o)[2])[2]/dq_m**2.
            
            # c_d=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
            
            
            #just to see what happen Hisashi
            # rho=rho_o-0.5*dt*(div_flux+div_flux_o)/2.
            rho=rho_o - dt*(div_flux+div_flux_o)/2.
            
            #================================================#
            
            div_flux_o=div_f(tn_o*wp_o,tn_o*wf_o,tn_o*wq_o, dp_m,df_m,dq_m)        
            div_flux=div_f(tn*wp,tn*wf,tn*wq, dp_m,df_m,dq_m)        
            div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)
            
            # d2_tn= gradient(gradient(tn_o)[0])[0]/dp_m**2.\
            #       +gradient(gradient(tn_o)[1])[1]/df_m**2.\
            #       +gradient(gradient(tn_o)[2])[2]/dq_m**2.
                  
            d2_tn = (
                gradient(gradient(tn_o, axis=0), axis=0) / dp_m**2 +
                gradient(gradient(tn_o, axis=1), axis=1) / df_m**2 +
                gradient(gradient(tn_o, axis=2), axis=2) / dq_m**2
                    )
            
            c_d=exp(-0.5*abs(lambda_c*d2_tn*dt/tn_o))
            
            tn=(tn_o - dt*c_d*((div_flux+div_flux_o)/2.\
                            +(gamma-1)*tn_o*div_w\
                            -0*lambda_c*d2_tn))
                
            # tn=(tn_o-0.5*dt*c_d*((div_flux+div_flux_o)/2.\
            #                 +(gamma-1)*tn_o*div_w\
            #                 -0*lambda_c*d2_tn))
    
    return (rho, tn)

# def AGW_AMP(gamma, mask_bound, w_o, w_m, dts, amb_v, n_sor):
#     # global wp,wf,wq
    
#     # deltas = [dp_m,df_m,dq_m]
    
#     pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    
#     wp_m, wq_m, wf_m = w_m[0], w_m[1], w_m[2]
#     wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
#     dp_m, dq_m, df_m, dt, dTin = dts[0], dts[1], dts[2], dts[3], dts[4]
#     # dp_m, dq_m, df_m, dt = dts[0], dts[1]*2.5, dts[2]*2.5, dts[3]
    
#     wpozero = wp_o[0,:,:]
    
#     wp, wf, wq = wp_o, wf_o, wq_o
#     rho,tn = rho_o,tn_o
    
#     #+++++++++++++++++++++++++++
#     pn=r_g*rho*tn
#     c_s=sqrt(gamma*pn/rho);  
    
#     lambda_c=c_s**2./nu_nn
    
#     wp_0 = 2*wp_o - wp_m;
#     wf_0 = 2*wf_o - wf_m;
#     wq_0 = 2*wq_o - wq_m;
        
#     for i_sor in range (0,n_sor):
        
#         wp[-1, :, :] = 0
            
#         # lambda_mu=pn/nu_nn
    
#         #=====================================================================#
        

        
#         #=====================================================================#
        
#         # wp_nl=w_nl(wp_o,wp_m)
#         # wf_nl=w_nl(wf_o,wf_m)
#         # wq_nl=w_nl(wq_o,wq_m)
        
#         #=====================================================================#
        
#         # div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
#         # div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
#         div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
#         div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
#         gr_pn=gradient(pn) #C
#         w_gr_pn=wp*gr_pn[0]/dp_m + wf*gr_pn[1]/df_m + wq*gr_pn[2]/dq_m #D
    
#         #=====================================================================#
        
        
#         # kherani 2012 book chapter
#         awip, awif, awiq = AWI(gamma, pn, div_w, dts, rho)
#         # awiip, awiif, awiiq = AWII(gamma, pn, div_w, dts, rho) #this term does exists in kherani 2012
#         gwip, gwif, gwiq = GWI(pn, div_flux, dts, rho)
#         gwiip, gwiif, gwiiq = GWII(w_gr_pn, dts, rho)
        
#         rtp, rtf, rtq = RTgr(dTin, dt, rho, dts)
        
#         # f_p=AGW_DAMP(0,gamma,dp_m,pn,rho,div_w,div_flux,w_gr_pn)
#         # f_f=AGW_DAMP(1,gamma,df_m,pn,rho,div_w,div_flux,w_gr_pn)
#         # f_q=AGW_DAMP(2,gamma,dq_m,pn,rho,div_w,div_flux,w_gr_pn) 
        
#         #=====================================================================#
#         wp_o[0,:,:] = wpozero
#         f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_amb,rho_o,tn_o, nu_nn,dp_m,dq_m,df_m,dt)
#         wp_damp=f[0];w_damp=f[1]
        
#         # d2_w=    gradient(gradient(wp_o)[0])[0]/dp_m**2.\
#         #         +gradient(gradient(wf_o)[1])[1]/df_m**2.\
#         #         +gradient(gradient(wq_o)[2])[2]/dq_m**2.
        
#         # c_visc=exp(-1.*abs((lambda_mu/rho)*d2_w*t/max(10,wp_o.max())))
#         # c_diff=1.#ndf(t,0,10)#*c_visc 
        
       
#         # awip,  awif,  awiq  = awip*0,  awif*0,  awiq*0 
#         # awiip, awiif, awiiq = awiip*0, awiif*0, awiiq*0 
#         # gwip,  gwif,  gwiq  = gwip*0,  gwif*0,  gwiq*0 
#         # gwiip, gwiif, gwiiq = gwiip*0, gwiif*0, gwiiq*0 
#         # rtp, rtf, rtq = rtp*0, rtf*0, rtq*0
        
#         # wp=wp_damp*(wp_0 + dt**2.*(awip + awiip + gwip + gwiip - rtp))
#         # wf= w_damp*(wf_0 + dt**2.*(awif + awiif + gwif + gwiif - rtf))
#         # wq= w_damp*(wq_0 + dt**2.*(awiq + awiiq + gwiq + gwiiq - rtq))
        
#         temp_term = [rtp, rtf, rtq]
        
#         wp=wp_damp*(wp_0 + dt**2.*(awip + gwip + gwiip - 0*rtp))*mask_bound
#         wf= w_damp*(wf_0 + dt**2.*(awif + gwif + gwiif - 0*rtf))*mask_bound
#         wq= w_damp*(wq_0 + dt**2.*(awiq + gwiq + gwiiq - 0*rtq))*mask_bound
        
#         # wp[:, [0, -1], :] = 0   # First and last row of each 2D slice
#         # wp[:, :, [0, -1]] = 0   # First and last column of each 2D slice
#         # wp[-1, :, :] = 0   # First and last depth layer
        
#         # wf[:, [0, -1], :] = 0   # First and last row of each 2D slice
#         # wf[:, :, [0, -1]] = 0   # First and last column of each 2D slice
#         # wf[-1, :, :] = 0   # First and last depth layer
        
#         # wq[:, [0, -1], :] = 0   # First and last row of each 2D slice
#         # wq[:, :, [0, -1]] = 0   # First and last column of each 2D slice
#         # wq[-1, :, :] = 0   # First and last depth layer
        
#         # wp[0,:,:]=wp_o[0,:,:]
#         # wf[0,:,:]=wf_o[0,:,:]
#         # wq[0,:,:]=wq_o[0,:,:]
        
#         # wp[:,:,:]=wp_o[:,:,:]
#         # wf[:,:,:]=wf_o[:,:,:]
#         # wq[:,:,:]=wq_o[:,:,:]

#         #Here appears that the program solves the temperature and pressure 
#         #=====================================================================#
    
    # for i_sor in range (0,n_sor):
        
    #     div_flux_o=div_f(0*rho_o*wp_o,rho_o*wf_o,rho_o*wq_o,dp_m,df_m,dq_m) #non0
    #     div_flux  =div_f(0*rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #non 0
        
    #     # d2_rho= gradient(gradient(rho_o)[0])[0]/dp_m**2.\
    #     #        +gradient(gradient(rho_o)[1])[1]/df_m**2.\
    #     #        +gradient(gradient(rho_o)[2])[2]/dq_m**2.
        
    #     # c_d=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
        
        
    #     #just to see what happen Hisashi
    #     # rho=rho_o-0.5*dt*(div_flux+div_flux_o)/2.
    #     rho=rho_o - 0*dt*(div_flux+div_flux_o)/2.
        
    #     #================================================#
        
    #     div_flux_o=div_f(tn_o*wp_o,tn_o*wf_o,tn_o*wq_o, dp_m,df_m,dq_m)        
    #     div_flux=div_f(tn*wp,tn*wf,tn*wq, dp_m,df_m,dq_m)        
    #     div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)
        
    #     d2_tn= gradient(gradient(tn_o)[0])[0]/dp_m**2.\
    #           +gradient(gradient(tn_o)[1])[1]/df_m**2.\
    #           +gradient(gradient(tn_o)[2])[2]/dq_m**2.
        
    #     c_d=exp(-0.5*abs(lambda_c*d2_tn*dt/tn_o))
        
    #     tn=(tn_o-0*dt*c_d*((div_flux+div_flux_o)/2.\
    #                     +(gamma-1)*tn_o*div_w\
    #                     -0*lambda_c*d2_tn))
            
    #     # tn=(tn_o-0.5*dt*c_d*((div_flux+div_flux_o)/2.\
    #     #                 +(gamma-1)*tn_o*div_w\
    #     #                 -0*lambda_c*d2_tn))
    
    # return (wp,wf,wq, rho,tn, temp_term)
#=============================================================================#    


def AGW_AMP_OLD(gamma, mask_bound, w_o, w_m, dts, 
                  pn_amb, rho_amb, rho_o, r_g, tn_o, dTin, nu_nn, \
                  i_t, n_sor):
    # global wp,wf,wq
    
    # deltas = [dp_m,df_m,dq_m]
    
    # pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn = amb_v[0], amb_v[1], amb_v[2], amb_v[3], amb_v[4], amb_v[5]
    
    wp_m, wq_m, wf_m = w_m[0], w_m[1], w_m[2]
    wp_o, wq_o, wf_o = w_o[0], w_o[1], w_o[2]
    dp_m, dq_m, df_m, dt = dts[0], dts[1], dts[2], dts[3]
    # dp_m, dq_m, df_m, dt = dts[0], dts[1]*2.5, dts[2]*2.5, dts[3]
    
    wp, wf, wq = wp_o, wf_o, wq_o
    rho,tn = rho_o,tn_o
    
    for i_sor in range (0,n_sor):
            
        pn=r_g*rho*tn
        c_s=sqrt(gamma*pn/rho);  
        
        lambda_c=c_s**2./nu_nn
        # lambda_mu=pn/nu_nn
    
        #=====================================================================#
        
        wp_0 = 2*wp_o - wp_m;
        wf_0 = 2*wf_o - wf_m;
        wq_0 = 2*wq_o - wq_m;
        
        #=====================================================================#
        
        # wp_nl=w_nl(wp_o,wp_m)
        # wf_nl=w_nl(wf_o,wf_m)
        # wq_nl=w_nl(wq_o,wq_m)
        
        #=====================================================================#
        
        # div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        # div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)  #A     
        div_flux=div_f(rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #B
        
        gr_pn=gradient(pn) #C
        w_gr_pn=wp*gr_pn[0]/dp_m + wf*gr_pn[1]/df_m + wq*gr_pn[2]/dq_m #D
    
        #=====================================================================#
        
        
        # kherani 2012 book chapter
        awip, awif, awiq = AWI(gamma, pn, div_w, dts, rho)
        # awiip, awiif, awiiq = AWII(gamma, pn, div_w, dts, rho) #this term does exists in kherani 2012
        gwip, gwif, gwiq = GWI(pn, div_flux, dts, rho)
        gwiip, gwiif, gwiiq = GWII(w_gr_pn, dts, rho)
        
        rtp, rtf, rtq = RTgr(dTin, dt, rho, dts)
        
        # f_p=AGW_DAMP(0,gamma,dp_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_f=AGW_DAMP(1,gamma,df_m,pn,rho,div_w,div_flux,w_gr_pn)
        # f_q=AGW_DAMP(2,gamma,dq_m,pn,rho,div_w,div_flux,w_gr_pn) 
        
        #=====================================================================#
        
        f=dissip_coef(mask_bound,wp_o,wq_o,wf_o, pn_amb,rho_amb,rho_o,tn_o, nu_nn,dp_m,dq_m,df_m,dt)
        wp_damp=f[0];w_damp=f[1]
        
        # d2_w=    gradient(gradient(wp_o)[0])[0]/dp_m**2.\
        #         +gradient(gradient(wf_o)[1])[1]/df_m**2.\
        #         +gradient(gradient(wq_o)[2])[2]/dq_m**2.
        
        # c_visc=exp(-1.*abs((lambda_mu/rho)*d2_w*t/max(10,wp_o.max())))
        # c_diff=1.#ndf(t,0,10)#*c_visc 
        
       
        # awip,  awif,  awiq  = awip*0,  awif*0,  awiq*0 
        # awiip, awiif, awiiq = awiip*0, awiif*0, awiiq*0 
        # gwip,  gwif,  gwiq  = gwip*0,  gwif*0,  gwiq*0 
        # gwiip, gwiif, gwiiq = gwiip*0, gwiif*0, gwiiq*0 
        # rtp, rtf, rtq = rtp*0, rtf*0, rtq*0
        
        # wp=wp_damp*(wp_0 + dt**2.*(awip + awiip + gwip + gwiip - rtp))
        # wf= w_damp*(wf_0 + dt**2.*(awif + awiif + gwif + gwiif - rtf))
        # wq= w_damp*(wq_0 + dt**2.*(awiq + awiiq + gwiq + gwiiq - rtq))
        
        temp_term = [rtp, rtf, rtq]
        
        wp=wp_damp*(wp_0 + dt**2.*(awip + gwip + gwiip - rtp))
        wf= w_damp*(wf_0 + dt**2.*(awif + gwif + gwiif - 0*rtf))
        wq= w_damp*(wq_0 + dt**2.*(awiq + gwiq + gwiiq - 0*rtq))
        
        # wp[:, [0, -1], :] = 0   # First and last row of each 2D slice
        # wp[:, :, [0, -1]] = 0   # First and last column of each 2D slice
        # wp[-1, :, :] = 0   # First and last depth layer
        
        # wf[:, [0, -1], :] = 0   # First and last row of each 2D slice
        # wf[:, :, [0, -1]] = 0   # First and last column of each 2D slice
        # wf[-1, :, :] = 0   # First and last depth layer
        
        # wq[:, [0, -1], :] = 0   # First and last row of each 2D slice
        # wq[:, :, [0, -1]] = 0   # First and last column of each 2D slice
        # wq[-1, :, :] = 0   # First and last depth layer
        
        # wp[0,:,:]=wp_o[0,:,:]
        # wf[0,:,:]=wf_o[0,:,:]
        # wq[0,:,:]=wq_o[0,:,:]
        
        # wp[:,:,:]=wp_o[:,:,:]
        # wf[:,:,:]=wf_o[:,:,:]
        # wq[:,:,:]=wq_o[:,:,:]

        #Here appears that the program solves the temperature and pressure 
        #=====================================================================#
    
    for i_sor in range (0,n_sor):
        
        div_flux_o=div_f(0*rho_o*wp_o,rho_o*wf_o,rho_o*wq_o,dp_m,df_m,dq_m) #non0
        div_flux  =div_f(0*rho*wp,rho*wf,rho*wq,dp_m,df_m,dq_m) #non 0
        
        # d2_rho= gradient(gradient(rho_o)[0])[0]/dp_m**2.\
        #        +gradient(gradient(rho_o)[1])[1]/df_m**2.\
        #        +gradient(gradient(rho_o)[2])[2]/dq_m**2.
        
        # c_d=exp(-0.*abs(lambda_c*d2_rho*t/rho_o))
        
        
        #just to see what happen Hisashi
        # rho=rho_o-0.5*dt*(div_flux+div_flux_o)/2.
        rho=rho_o - 0*dt*(div_flux+div_flux_o)/2.
        
        #================================================#
        
        div_flux_o=div_f(tn_o*wp_o,tn_o*wf_o,tn_o*wq_o, dp_m,df_m,dq_m)        
        div_flux=div_f(tn*wp,tn*wf,tn*wq, dp_m,df_m,dq_m)        
        div_w=div_f(wp,wf,wq,dp_m,df_m,dq_m)
        
        d2_tn= gradient(gradient(tn_o)[0])[0]/dp_m**2.\
              +gradient(gradient(tn_o)[1])[1]/df_m**2.\
              +gradient(gradient(tn_o)[2])[2]/dq_m**2.
        
        c_d=exp(-0.5*abs(lambda_c*d2_tn*dt/tn_o))
        
        tn=(tn_o-0*dt*c_d*((div_flux+div_flux_o)/2.\
                        +(gamma-1)*tn_o*div_w\
                        -0*lambda_c*d2_tn))
            
        # tn=(tn_o-0.5*dt*c_d*((div_flux+div_flux_o)/2.\
        #                 +(gamma-1)*tn_o*div_w\
        #                 -0*lambda_c*d2_tn))
    
    return (wp,wf,wq, rho,tn, temp_term)