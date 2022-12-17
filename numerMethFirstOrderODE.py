import numpy as np
import pandas as pd
# Thomas Höft, University of St. Thomas
# hoft@stthomas.edu
# license: Attribution-NonCommercial-ShareAlike 4.0 International 
# https://creativecommons.org/licenses/by-nc-sa/4.0/
def EulerMethod(t_0, y_0, del_t, n, f):
    y = np.zeros(n+1)
    y[0] = y_0
    t = np.linspace(t_0, t_0+n*del_t, n+1)
    for i in range(1,n+1):
        y[i] = y[i-1] + f( t[i-1], y[i-1] ) * del_t
    return (y)
def ImpEulerMethod(t_0, y_0, del_t, n, f):
    y = np.zeros(n+1)
    y[0] = y_0
    t = np.linspace(t_0, t_0+n*del_t, n+1)
    for i in range(1,n+1):
        k1 = f(t[i-1], y[i-1]           ) # predictor
        k2 = f(t[i  ], y[i-1] + k1*del_t) # corrector
        y[i] = y[i-1] + (k1+k2)/2 * del_t # Euler step
    return (y)
def RK4Method(t_0, y_0, del_t, n, f):
    y = np.zeros(n+1)
    y[0] = y_0
    t = np.linspace(t_0, t_0+n*del_t, n+1)
    for i in range(1,n+1):
        k1 = f(t[i-1]          , y[i-1]             )
        k2 = f(t[i-1] + del_t/2, y[i-1] + k1*del_t/2)
        k3 = f(t[i-1] + del_t/2, y[i-1] + k2*del_t/2)
        k4 = f(t[i-1] + del_t  , y[i-1] + k3*del_t  )
        y[i] = y[i-1] + (k1+2*k2+2*k3+k4)/6 * del_t # Euler step
    return (y)

var('t, y')
@interact#(layout=dict(top=[['y_prime_str'], 
         #                  ['y_0', 't_0', 't_n'], 
         #                  ['n'],
         #                  ['eul_bool', 'imp_bool', 'rk4_bool'],
         #                  ['ext_bool', 'y_exact_str']] ) )
def euler_method(
        y_prime_str = input_box('y*(1-y)', type=str, label="dy/dt = ", width=40), 
        y_0 = input_box(0.1, label='y(t_0) = ', width=10), 
        t_0 = input_box(  0, label='t_0 = ', width=10), 
        t_n = input_box( 10, label='t_n = ', width=10), 
        #n = slider([2^m for m in range(0,10)], default=8, label='# steps: '),
        n = slider([1,2,4,8,16,32,64,128,512], default=8, label='# steps: '),
        #n = input_box(2, label='(# steps) n = ', width=10),
        eul_bool = checkbox(True, label="Euler's Method"), 
        imp_bool = checkbox(True, label="Improved Euler"), 
        rk4_bool = checkbox(True, label="RK4"), 
        ext_bool = checkbox(False, label="Use exact solution"),
        y_exact_str = input_box('1/(1+9*exp(-t))', type=str, label='y(t) = ', width=40), 
        ):

    ww_bool = False # print values & errors on one line to enter into e.g. WeBWorK
    #np.set_printoptions(formatter={'float': '{:G}'.format}) # numpy arrays but not numbers
    print("\n Reminder: use * for multiplication, ** for powers, and exp(...)",
          "for the exponential function. \n")
    
    y_exact_fn = lambda t: eval(y_exact_str)
    y_prime_fn = lambda t,y: eval(y_prime_str)
        
    cols = ['step', 't'] # printing data header column 

    del_t = float((t_n-t_0)/n) # timestep
    ts = np.linspace(t_0, t_n, n+1) # t-values
    inds = np.arange(len(ts)) # step indices for printing
    sols = np.array([inds, ts]) # start solutions array
    errs = np.array([inds, ts]) # start error array
    if ext_bool: y_true = np.asarray([ y_exact_fn(tp) for tp in ts ])
    #y_true = [ y_exact_fn(tp) for tp in ts] if ext_bool else [np.zeros_like(ts)]
    
    y_plt = plot([]) # initialize empty plot
    if eul_bool:
        y_eul = EulerMethod(t_0, y_0, del_t, n, y_prime_fn)
        sols = np.vstack((sols, y_eul))
        if ext_bool: err_eul = np.abs(y_true-y_eul)
        if ext_bool: errs = np.vstack((errs, np.abs(y_true-y_eul)))
        cols.append('Euler')
        eul_plot = line(zip(ts,y_eul), rgbcolor=(1/4,1/4,3/4), marker='.', 
                        linestyle='-.', legend_label='Euler')
        y_plt += eul_plot
    else: y_eul = np.zeros_like(ts)
    if imp_bool:
        y_imp = ImpEulerMethod(t_0, y_0, del_t, n, y_prime_fn)
        sols = np.vstack((sols, y_imp))
        if ext_bool: err_imp = np.abs(y_true-y_imp)
        if ext_bool: errs = np.vstack((errs, np.abs(y_true-y_imp)))
        cols.append('Imp. Euler')
        imp_plot = line(zip(ts,y_imp), rgbcolor=(1/4,3/4,1/8), marker='.', 
                          linestyle=':',  legend_label='Improved Euler')
        y_plt += imp_plot
    else: y_imp = np.zeros_like(ts)
    if rk4_bool:
        y_rk4 = RK4Method(t_0, y_0, del_t, n, y_prime_fn)
        sols = np.vstack((sols, y_rk4))
        if ext_bool: err_rk4 = np.abs(y_true-y_rk4)
        if ext_bool: errs = np.vstack((errs, np.abs(y_true-y_rk4)))
        cols.append('RK4')
        rk4_plot = line(zip(ts,y_rk4), rgbcolor=(1  ,1/4,1/2), marker='.', 
                        linestyle='--', legend_label='RK4')
        y_plt += rk4_plot
    else: y_rk4 = np.zeros_like(ts)
    if ext_bool:
        cols.append('True soln')
        sols = np.vstack((sols, y_true))
        exact_plot = plot(y_exact_fn(x), t_0, t_n, rgbcolor=(0,0,0), 
                          legend_label='Exact', thickness=2)
        y_plt += exact_plot
    
    # figure out y range for slope field
    if sols[2:,:].size: # not empty
        y_max = max(sols[2:,:].ravel())
        y_min = min(sols[2:,:].ravel())
    else:
        y_max=1
        y_min=-0.5
    y_mid = (y_max+y_min)/2
    y_r = (y_max-y_min)/2
    if y_r==0: y_r = 1
    y_max = y_mid + y_r*1.2
    y_min = y_mid - y_r*1.2
    if (y_max,y_min)==(0,0): # maybe not necessary?...
        y_max=1
        y_min=-0.5
    if np.isnan(y_max) or np.isinf(y_max): y_max = 1
    if np.isnan(y_min) or np.isinf(y_min): y_min = -0.5

    slopes = plot_slope_field(y_prime_fn(t,y), (t,t_0,t_n), 
                              (y,y_min, y_max), color='lightgray')
    y_plt += slopes
        
    show(y_plt, axes_labels=['t', 'y'], legend_loc="lower right", ymin=y_min, ymax=y_max)

    # put into DataFrame for convenient printing to screen
    pd.options.display.float_format = '{:#0.9G}'.format
    df_y = pd.DataFrame(sols.T, columns=cols) 
    df_y = df_y.astype({'step':int})
    df_y = df_y.set_index('step')
    if ext_bool:
        df_e = pd.DataFrame(errs.T, columns=cols[0:-1])
        df_e = df_e.astype({'step':int})
        df_e = df_e.set_index('step')

    sep = 80*"-"
    print(sep)
    print()
    print("dy/dt = ", y_prime_str)
    print("y({}) = {}".format(t_0, y_0))
    print("y({}) = ?".format(t_n))
    print()
    print("n = ", n)
    print("delta t = ", del_t)
    print()
    if ext_bool:
        print("y(t) = ", y_exact_str)
        print()
    print(sep)
    print() # could've done next little bit w/ numpy arrays instead of DataFrame...
    if ext_bool: print("            y(t_n) = {:#0.9G}".format(float(df_y["True soln"].iloc[[-1]])))
    if eul_bool: print("         Euler y_n = {:#0.9G}".format(float(df_y["Euler"].iloc[[-1]])))
    if imp_bool: print("Improved Euler y_n = {:#0.9G}".format(float(df_y["Imp. Euler"].iloc[[-1]])))
    if rk4_bool: print("           RK4 y_n = {:#0.9G}".format(float(df_y["RK4"].iloc[[-1]])))
    if ext_bool:
        print()
        if eul_bool: print("         Euler e_n = {:#0.9G}".format(float(df_e["Euler"].iloc[[-1]])))
        if imp_bool: print("Improved Euler e_n = {:#0.9G}".format(float(df_e["Imp. Euler"].iloc[[-1]])))
        if rk4_bool: print("           RK4 e_n = {:#0.9G}".format(float(df_e["RK4"].iloc[[-1]])))
    print()
    print(sep)
    if ww_bool:
        print()
        print('Formatted for entering into WeBWorK:')
        print()
        if eul_bool: print("Euler values:"         , df_y["Euler"     ].to_string(index=False, header=False).replace('\n',' '))
        if imp_bool: print("Improved Euler values:", df_y["Imp. Euler"].to_string(index=False, header=False).replace('\n',' '))
        if rk4_bool: print("RK4 values:",            df_y["RK4"       ].to_string(index=False, header=False).replace('\n',' '))
        print()
        if ext_bool:
            if eul_bool: print("Euler errors:"         , df_e["Euler"     ].to_string(index=False, header=False).replace('\n',' '))
            if imp_bool: print("Improved Euler errors:", df_e["Imp. Euler"].to_string(index=False, header=False).replace('\n',' '))
            if rk4_bool: print("RK4 errors:"           , df_e["RK4"       ].to_string(index=False, header=False).replace('\n',' '))
            print()
        print(sep)
    print()
    print("Values:")
    print(df_y.to_string())
    if ext_bool & any((eul_bool, imp_bool, rk4_bool)):
        print()
        print("Errors:")
        print(df_e.to_string())
    print()
    print(sep)
