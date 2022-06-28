import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.integrate import solve_ivp
from scipy.integrate import simpson

warnings.filterwarnings("ignore")

def deqs(r,y,g):  #Define DEQS
    
    return [y[1]/r**2,-r**2*y[0]**(1/(g-1))]

def stop(r,y,g): return y[0] #Define Integration Stop

stop.terminal = True

tmin=1e-4 #Lower time time span 
tmax=20 #Upper time span
y0=np.array([1,0]) #Initial Conditions


g_mat=np.linspace(1.25,1.7,100) #Gamma matrix
S=[] #Stores configurational entropy values for each gamma value
S1=[] #Stores Sa^3 values for each gamma value and 1.00π/R
S2=[] #Stores Sa^3 values for each gamma value and 0.95π/R
S3=[] #Stores Sa^3 values for each gamma value and 1.05π/R
M=[] #Stores mass values for each gamma value
g1_mat=np.array([1.2,1.4,1.7]) #Gamma values for which we plot the first figure


for g in g1_mat:   
   
    sol = solve_ivp(deqs,[tmin,tmax],y0,method='RK45',args=(g,),atol=1e-8,rtol=1e-8,events=stop)                #Solve Lane Emden
    kmin = (np.pi/sol.t[-1])                                                                                    #Define kmin
    k = np.linspace(kmin,100*kmin,10000)                                                                        #Initialize k vector
    hkmin = (simpson(sol.y[0]**(1/(g-1))*np.sin(kmin*sol.t)*sol.t,sol.t,dx=0.001)*(1/kmin))**2                  #find h(kmin)
    hk_mat = np.zeros(10000)                                                                                    #Initialize matrix h(k) 
    for i1,k1 in enumerate(k):
        hk_mat[i1] = (simpson(sol.y[0]**(1/(g-1))*np.sin(k1*sol.t)*sol.t,sol.t,dx=0.001)*(1/k1))**2             #find h(k) for each k 
    f = hk_mat/hkmin                                                                                            #find normalized f 
    label1="γ="+str(g)
    plt.plot(k/np.sqrt(g/(g-1)),f,label=label1) #Plot first figure
    
#Figure Configuration
plt.xlim([0,1.5])
plt.ylabel(r"$ f(|k|)$")
plt.xlabel(r"$k/\sqrt{4\pi G/K}$")
plt.legend()


for g in g_mat:
    sol = solve_ivp(deqs,[tmin,tmax],y0,method='RK45',args=(g,),atol=1e-12,rtol=1e-12,events=stop)  #Solve Lane Emden for each gamma
    hk_mat = np.zeros(10000)
    kmin = (np.pi/sol.t[-1]) #Define kmin for each gamma
    k = np.linspace(kmin,20*kmin,10000)
    hkmin = (simpson(sol.y[0]**(1/(g-1))*np.sin(kmin*sol.t)*sol.t,sol.t,dx=0.001)*(1/kmin))**2 #find h(kmin)
    
    for i1,k1 in enumerate(k):
        hk_mat[i1] = (simpson(sol.y[0]**(1/(g-1))*np.sin(k1*sol.t)*sol.t,sol.t,dx=0.001)*(1/k1))**2 #Find h(k) for each k
    
    #Repeat same process for 0.95π/R and 1.05π/R for figure 3
    hk_mat1 = np.zeros(10000)
    kmin1=(np.pi/(0.95*sol.t[-1]))
    k11=np.linspace(kmin1,20*kmin1,10000)
    hkmin1 = (simpson(sol.y[0]**(1/(g-1))*np.sin(kmin1*sol.t)*sol.t,sol.t,dx=0.001)*(1/kmin1))**2
    
    for i2,k2 in enumerate(k11):
        hk_mat1[i2] = (simpson(sol.y[0]**(1/(g-1))*np.sin(k2*sol.t)*sol.t,sol.t,dx=0.001)*(1/k2))**2
    
    hk_mat2 = np.zeros(10000)
    kmin2=(np.pi/(1.05*sol.t[-1]))
    k22=np.linspace(kmin2,20*kmin2,10000)    
    hkmin2= (simpson(sol.y[0]**(1/(g-1))*np.sin(kmin2*sol.t)*sol.t,sol.t,dx=0.001)*(1/kmin2))**2
    
    for i3,k3 in enumerate(k22):
        hk_mat2[i3] = (simpson(sol.y[0]**(1/(g-1))*np.sin(k3*sol.t)*sol.t,sol.t,dx=0.001)*(1/k3))**2
        
    f = hk_mat/hkmin
    f1 =hk_mat1/hkmin1
    f2 =hk_mat2/hkmin2
    S.append(-4*np.pi*((g/(g-1)))**(-3/2)*simpson(f*np.log(f)*k**2,k,dx=0.001))
    S1.append(-4*np.pi*simpson(f*np.log(f)*k**2,k,dx=0.001))
    S2.append(-4*np.pi*simpson(f1*np.log(f1)*k11**2,k11,dx=0.001))
    S3.append(-4*np.pi*simpson(f2*np.log(f2)*k22**2,k22,dx=0.001))
    M.append(((4*np.pi)*((g/(g-1))**((3/2)))*simpson((sol.y[0]**(1/(g-1)))*sol.t**2,sol.t,dx=0.001))/200)
  
    
  
    
plt.figure()
plt.plot(g_mat,M,'|',color='g',label=r'$M/\left( 200\left( \frac{K}{4\pi G}\right)^{\frac{3}{2}} \rho_c^{\frac{3}{2}\gamma -2} \right)$')
plt.plot(g_mat,S,color='r',label=r'$S\rho_0^{-1}/\left( \left( \frac{K}{4\pi G}\right)^{-\frac{3}{2}} \rho_c^{2-\frac{3}{2}\gamma} \right)$')
plt.legend()
plt.xlabel("γ")

plt.figure()
plt.plot(g_mat,S1,color='black',label='π/(1.00R)')
plt.plot(g_mat,S2,'|',color='blue',label='π/(0.95R)')
plt.plot(g_mat,S3,'-.',color='red',label='π/(1.05R)')
plt.legend()
plt.axvline(4/3,linestyle='--',color='grey')
plt.axvline(5/3,linestyle='--',color='grey')
plt.xlabel("γ")
plt.ylabel(r"$S\alpha^{3}$")
plt.legend()