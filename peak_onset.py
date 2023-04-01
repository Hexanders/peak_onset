import xarray as xr
from typing import Dict
import numpy as np
import array
import vam.whittaker as wtk 
import matplotlib.pyplot as plt
from arpes.fits.fit_models import AffineBroadenedFD, QuadraticModel, LinearModel 

def fit_c60_homo(data, 
                 smooth = False,
                 lmbd  = False,
                 asym = False,
                 evaluate_smooth = False,
                 calc_smoothing_error = False,
                 smooth_err_asym_min = 0.1,
                 smooth_err_asym_max = 0.9,
                 smoothing_error_plot = False,
                 interpolate_befor_ableiten = False,
                 Lleft = -10,
                 Lright = 10,
                 Lstep = 1000,
                 show_plot = False,
                 bkg_slope = None,
                 bkg_center = None,
                 flank_range_factor = 2., 
                 bkg_range_factor = 2.):
    """
    Estimate position of HOMO level from ups data slice. 
    By Estimating of the peak (minimum of second derivative) 
    and inflaction point (minimum of first derivative). 
    Difference of thous points is the fit area center on eV of inflacktion point 
    and eV of background where the first derivative gets positive agin, so slope change.
    
    By fitting of two lines this two areas and finding the intersection point you get you HOMO energy
    
    Parameters:
        data:                   [pyarpes xarray] slice integrated over phi 
        smooth:                 [boolean] if true calculate smoothing after Whittaker-Eilers
        evaluet_smooth:         [boolean] if true the tangente and background would be search in smoothed data nad not in original data
        calc_smoothing_error:   [boolean] if true calculate smoothing of anveloupe (max/min smoothing) and finde intersection with error, could be time consuming  
        Lleft:                  [int] left boarder for penalty lambda in Whittaker-Eilers smoothing
        Lright:                 [int] right boarder ....
        Lstep:                  [int] step for lambda range in Whittaker-Eilers smoothing
        show_plot:              [boolean] If True plot data and ddifferentials in separate plots
        bkg_center:             [float or None] if not None center position for background fitting 
        flank_range_factor:     [float] width of the range for flank fitting respective to distance from peak to inflaction point. 
                                         E.g. 2. means whole width, 4. half width
        bkg_range_factor:       [float] same as flank_range_factor but for background
    
    Returns:
        [dict] 'E_homo':       HOMO level energy position 
               'tangente_fit': lmfit paramas for fit of the flank of peak  
               'bkg_fit':      lmfit paramas for fit of the background  
               'd/dx':         energy position of minimu of first derivatef of data 
               'd/dx/dx':      energy position of minimu of second derivatef of data
    """
    if smooth == True:
        smooth_data, first_d, second_d = whittaker_eilers_smoother(data, 
                                                                   lmbd = lmbd,
                                                                   asym = asym,
                                                                   Lleft = Lleft, 
                                                                   Lright = Lright, 
                                                                    Lstep = Lstep)
        ### Calculate Smoothing error from envelope calculation for max an min posible smoothong, so to say
        
        # if calc_smoothing_error:
        #     asym_range = np.arange(0.5,0.95,0.05)
        #     asym_error_results = {}
        #     for asym_i in asym_range:
        #         current_smoothing = whittaker_eilers_smoother(data, 
        #                                                lmbd = lmbd,
        #                                                asym = asym_i,
        #                                                Lleft = Lleft, 
        #                                                Lright = Lright, 
        #                                                Lstep = Lstep)
        #         c_smooth_f_d_eV, c_smooth_s_d_eV , c_smooth_dif , c_smooth_tang_roi , c_smooth_bkg_roi = find_inflaction(current_smoothing[0], 
        #                                                           current_smoothing[1], 
        #                                                            current_smoothing[2], 
        #                                                             bkg_center = bkg_center,
        #                                                             flank_range_factor = flank_range_factor, 
        #                                                             bkg_range_factor = bkg_range_factor)
        #         asym_error_results[asym_i] = find_intersec(c_smooth_tang_roi, c_smooth_bkg_roi)
    else:
        first_d = data.differentiate('eV',edge_order = 1)
        second_d = first_d.differentiate('eV',edge_order = 1)
        smooth_data = None # Just if you do nos smooth data, other wise output is corupted

            
    if evaluate_smooth: 
        eval_data = smooth_data
    else:
        eval_data = data
        
    if interpolate_befor_ableiten:
        xvals = np.linspace(min(eval_data.eV), max(eval_data.eV), len(eval_data.eV)*5)
        inter_first_d = np.interp(xvals, eval_data.eV ,eval_data.data)
        #from scipy.interpolate import CubicSpline
        #inter_first_d = CubicSpline(eval_data.eV, eval_data.data)
        #inter_first_d = inter_first_d(xvals)
        first_d = xr.DataArray(inter_first_d, 
                               dims = ['eV'],
                                name = 'spectrum',
                                coords={'eV': xvals}).differentiate('eV',edge_order = 1)
        
    first_d_eV, second_d_eV, dif, tangente_roi, bkg_roi = find_inflaction(eval_data,
                                                                     first_d, second_d,  
                                                                     bkg_center = bkg_center,
                                                                     flank_range_factor = flank_range_factor, 
                                                                     bkg_range_factor = bkg_range_factor)    
    
    tmp = find_intersec(tangente_roi, bkg_roi)
    x_intersection = tmp['intersection_eV']
    x_intersection_err = tmp['intersection_eV_err']
    tangente_fit = tmp['tangente_roi_fit']
    bkg_roi_fit = tmp['bkg_roi_fit']
    
    if calc_smoothing_error:
        smooth_err = smoothing_max_min(data, 
                        first_d_eV,
                        asym_min = smooth_err_asym_min,
                        asym_max = smooth_err_asym_max,
                        plot_results = smoothing_error_plot)
        inflaction_divation = []
        intersection_divation = []
        intersection_divation_err = []
        for ḱ,v in  smooth_err.items():
            c_smooth_f_d_eV, c_smooth_s_d_eV , c_smooth_dif , c_smooth_tang_roi , c_smooth_bkg_roi = find_inflaction(v[0], 
                                                                  v[1], 
                                                                   v[2], 
                                                                    bkg_center = bkg_center,
                                                                    flank_range_factor = flank_range_factor, 
                                                                    bkg_range_factor = bkg_range_factor)
            inflaction_divation.append(float(c_smooth_f_d_eV))
            tmp = find_intersec(c_smooth_tang_roi , c_smooth_bkg_roi)
            intersection_divation.append(float(tmp['intersection_eV']))
            intersection_divation_err.append(float(tmp['intersection_eV_err']))
            if smoothing_error_plot:
                plt.plot(data.eV.data, data.data, marker= '.', ls = '')
                plt.plot(data.eV.data, data.eV.data*tmp['tangente_roi_fit'].values['slope']+tmp['tangente_roi_fit'].values['intercept'])
                plt.plot(data.eV.data, data.eV.data*tmp['bkg_roi_fit'].values['slope']+tmp['bkg_roi_fit'].values['intercept'])
                differencien = abs(max(data.data)-min(data.data))*0.1 ## opticaly pleasing
                plt.ylim(min(data.data)-differencien,max(data.data)+differencien)
        uper_error = []
        del_uper_error = []
        lower_error = []
        del_lower_error = []
        for i,k in zip(intersection_divation,intersection_divation_err):
            if x_intersection - i > 0.:
                uper_error.append(i)
                del_uper_error.append(k)
            if x_intersection - i <= 0.:
                lower_error.append(i)
                del_lower_error.append(k)
        ## calculate average and error of average
        if uper_error:
            uper_error = np.average(uper_error)
            del_uper_error =  np.sqrt(np.sum([(i/len(del_uper_error))**2. for i in del_uper_error]))
            uper_error = np.sqrt((x_intersection - (uper_error + del_uper_error))**2. + x_intersection_err**2.) #merry smoothong error with fit errors
            #del_uper_error = np.sqrt(x_intersection_err**2. + del_uper_error**2.)
        else:
            uper_error = x_intersection_err
            del_uper_error = 0.
        if lower_error:
            lower_error = np.average(lower_error)
            del_lower_error =  np.sqrt(np.sum([(i/len(del_lower_error))**2. for i in del_lower_error]))
            lower_error = np.sqrt((x_intersection - (lower_error - del_lower_error))**2. + x_intersection_err**2.)  #merry smoothong error with fit errors
            #del_lower_error = np.sqrt(x_intersection_err**2. + del_lower_error**2.)
        else:
            lower_error = x_intersection_err
            del_lower_error = 0.
        
    if show_plot:
        fig,ax = plt.subplots(4,1,sharex=True)
        data.plot(ls = '', marker = '.', ax = ax[0], label ='Data')
        if smooth:
            f_d_label = "smooth d/dx"
            s_d_label = "smooth $(d/dx)^2$"
            #fdiv_smooth_data.plot(ls = '-', marker = '.', ax = ax[2], label = 'smooth')
        else:
            f_d_label = "d/dx"
            s_d_label = "$(d/dx)^2$"
        
        first_d.plot(ls = '-', marker = '.', ax = ax[2], 
                     label = f_d_label,
                     #label = '',
                     #color = 'k'
                    )
        second_d.plot(ls = '-', marker = '.', ax = ax[3], 
                      label = s_d_label,
                      #label ='',
                      #color = 'k'
                     )
        tangente_roi.plot(ls = '-', marker = 'o', ax = ax[0])
        bkg_roi.plot(ls = '-', marker = 'o', ax = ax[0])
        if smooth: 
            smooth_data.plot(ax = ax[0], label = 'smooth data')
            (data - smooth_data).plot(ax = ax[1], 
                                      label = 'diff: data - smooth data'
                                     )
        
        ax[0].plot(data.eV.data, data.eV.data*tangente_fit.values['slope']+tangente_fit.values['intercept'])
        ax[0].plot(data.eV.data, data.eV.data*bkg_roi_fit.values['slope']+bkg_roi_fit.values['intercept'])
        ax[0].vlines(x_intersection,data.data.min(),data.data.max())
        ax[0].set_ylim(data.data.min()-data.data.min()*0.1,data.data.max()+data.data.max()*0.1) #set limits of plots auf max min plus 10%
        ax[0].annotate(round(x_intersection,2), xy = (x_intersection,data.data.max()))
        #ax[1].plot(data.where(data.eV.data == tangente_roi).spectrum - (tangente_roi*tangente_fit.values['slope']+tangente_fit.values['intercept']) )  
        #data.differentiate('eV').differentiate('eV').plot()
        #plt.tight_layout()
        ylabels = ['Int. [a.u.]', 'Residual', 'd/dE','$(d/dE)^2$',]
        ax[0].legend()
        for a,yl in zip(ax,ylabels):
            a.set_title('')
            a.grid()
            a.set_xlabel('')
            a.set_ylabel(yl)            
            #a.legend()
        if smooth:
            plot_title = f'd/dx: {np.round(first_d_eV,2)}, d/dx2: {np.round(second_d_eV,2)},diff: {round(dif,2)}, $lambda$:{round(smooth_data.attrs["lambda"],2)}'
        else:
            plot_title = f'd/dx: {np.round(first_d_eV,2)}, d/dx2: {np.round(second_d_eV,2)},diff: {round(dif,2)}'
            
        ax[0].set_title(plot_title)
        plt.tight_layout()
        plt.show() 
    output = {}

    ### Colect data for output
    output['E_homo'] = x_intersection
    if calc_smoothing_error:
        output['E_homo_uper_err'] = uper_error
        output['del_uper_err'] = del_uper_error 
        output['E_homo_lower_err'] = lower_error 
        output['del_lower_err'] = del_lower_error
    output['intersection_err'] = x_intersection_err
    if calc_smoothing_error:
        output['intersection_divation'] = intersection_divation
        output['intersection_divation_err'] = intersection_divation_err            
    output['tangente_fit'] = tangente_fit
    output['bkg_fit'] = bkg_roi_fit
    output['inflection_point'] = float(first_d_eV)
    if calc_smoothing_error:
        output['inflaction_divation'] = inflaction_divation
        output['max_point'] = float(second_d_eV)
    output['smooth_data'] = smooth_data

    return output

def find_inflaction(data, 
                    first_d, 
                    second_d, 
                    bkg_center = None,
                    flank_range_factor = 2. , 
                    bkg_range_factor= 2.):
    """
    Find the inflaction point and background region for a given spectrum and its first and second derivatives.

    Args:
        data (xarray.Dataset): The full spectrum data with dimensions 'eV' and 'phi'.
        first_d (xarray.DataArray): The first derivative of the spectrum with respect to energy (dimension 'eV').
        second_d (xarray.DataArray): The second derivative of the spectrum with respect to energy (dimension 'eV').
        bkg_center (float, optional): The center position of the background region. If not provided, it will be determined based on the slope change of the first derivative. Default is None.
        flank_range_factor (float, optional): The factor to determine the range of the tangent region around the inflation point. Default is 2.0.
        bkg_range_factor (float, optional): The factor to determine the range of the background region around its center. Default is 2.0.

    Returns:
        A tuple with the following elements:
        - first_d_eV (float): The energy position of the inflation point.
        - second_d_eV (float): The energy position of the peak, (minimum) of the second derivative.
        - dif (float): The absolute difference between first_d_eV and second_d_eV.
        - tangente_roi (xarray.Dataset): A subset of the input data with the energy range around the inflation point determined by flank_range_factor.
        - bkg_roi (xarray.Dataset): A subset of the input data with the energy range around the background region determined by bkg_center and bkg_range_factor.
    """
    ### locate wendepoint
    first_d_eV = first_d.eV.where(first_d.data == first_d.data.min()).min().data # position of inflacktion point
    second_d_eV =second_d.eV.where(second_d.data == second_d.data.min()).min().data # position of peak

    dif = abs(first_d_eV-second_d_eV)
    tangente_roi = data.sel(eV = slice(first_d_eV-dif/flank_range_factor, first_d_eV + dif/flank_range_factor), 
                                  #phi = slice(-0.5,0.5)
                                  #phi = slice(3.3,7.5)
                                 )

    # print(first_d_eV, second_d_eV,tangente)
    ### locate right border of background
    if bkg_center:
        right_bkg_boarder = bkg_center
    else:
        right_bkg_boarder = first_d.eV.min().values

        for a in first_d:           # Finde first point alonde eV wehre first derivative is positiv. so Slope change this is background reagion
            if a.eV.item() > first_d_eV:
                if a.values > 0:
                    break
                else:
                    right_bkg_boarder = a.eV.item()
    bkg_roi = data.sel(eV = slice(right_bkg_boarder-dif/bkg_range_factor, right_bkg_boarder+dif/bkg_range_factor), 
                                  #phi = slice(-0.5,0.5)
                                  #phi = slice(3.3,7.5)
                                 )
    return float(first_d_eV), float(second_d_eV), dif, tangente_roi, bkg_roi
    
def whittaker_eilers_smoother(data,
                              lmbd = False,
                              asym = False, 
                              Lleft = -10,
                              Lright = 10,
                              Lstep = 1000):
    """
    Applies the Whittaker-Eilers smoother algorithm to smooth the given data.

    Parameters:
    -----------
    data: xarray.DataArray
        The data to be smoothed. It is expected that the data is a one-dimensional array with an 'eV' coordinate.

    lmbd: float or False, optional
        The smoothing parameter. If a float is provided, this value is used as the smoothing parameter. If False (default), 
        the smoothing parameter is determined by optimizing the smoothing using the Whittaker-Eilers algorithm.

    asym: float or False, optional
        The degree of asymmetry of the weights. If a float is provided, this value is used as the degree of asymmetry. 
        If False (default), symmetric weights are used.

    Lleft: float, optional
        The left endpoint of the range of smoothing parameters to be considered when optimizing lambda. Default is -10.

    Lright: float, optional
        The right endpoint of the range of smoothing parameters to be considered when optimizing lambda. Default is 10.

    Lstep: int, optional
        The number of steps to use in the grid search for optimizing lambda. Default is 1000.

    Returns:
    --------
    tuple
        A tuple of three xarray.DataArray objects:
        - The smoothed data, with the same shape as the input data.
        - The first derivative of the smoothed data, with the same shape as the input data.
        - The second derivative of the smoothed data, with the same shape as the input data.
    """
    y = data.data
    w = np.array((y!=-3000)*1,dtype='double')
    if lmbd:
        zv = wtk.ws2d(y, lmbd, w)
        loptv = lmbd
    else: 
        lrange = array.array('d',np.linspace(Lleft,Lright,Lstep))
        if asym:
            #asymtric weights 
            zv,loptv = wtk.ws2doptvp(y,w,lrange,p=asym)
        else:
            #otimize lambda
            zv,loptv = wtk.ws2doptv(y,w,lrange)
    smooth_data = data.copy(deep=True) 
    smooth_data.data = zv
    smooth_data.attrs ={'lambda':loptv}
    first_d = smooth_data.differentiate('eV',edge_order = 1)

    y2 = first_d.data
    w2 = np.array((y2!=-3000)*1,dtype='double')
    if lmbd:
        zv2 = wtk.ws2d(y2, lmbd, w2)
        loptv2 = lmbd 
    else:
        zv2,loptv2 = wtk.ws2doptv(y2,w2,lrange)
    fdiv_smooth_data = first_d.copy(deep=True) 
    fdiv_smooth_data.data = zv2
    fdiv_smooth_data.attrs ={'lambda':loptv2}
    second_d = fdiv_smooth_data.differentiate('eV',edge_order = 1)
    return smooth_data, fdiv_smooth_data, second_d



def smoothing_max_min(test_data: xr.Dataset, 
                    inflection_point: float,
                    asym_min: float = 0.1,
                    asym_max: float = 0.9, 
                    plot_results: bool = False) -> Dict[str, float]:

    """
    Apply Whittaker-Eilers smoothing to the input dataset and return two smoothed datasets:
    Split data in two at inflaction point. Smotth one siede with the maximal waigthing of 0.95 and another wit min waighting of 0.01.
    combine the data and smooth it again in order to avoid the sigularity at inflaction point. Than do it again but switsch the waitings for left and right sides.
    You get two smooth curvse with "maximal" and "minimal" smoothing like by max and min regration... Plot results for clarity 

    Parameters:
    -----------
    test_data: xr.Dataset
        The input dataset to smooth.
    inflection_point: float
        The inflection point to use to split the dataset in two.
        This point will be excluded from the smoothing process.
    asym_min: float, optional (default=0.01)
        The asymmetry parameter to use for the low-energy side of the dataset.
        This parameter controls the degree of smoothing and should be between 0 and 1.
        Smaller values lead to smoother results.
    asym_max: float, optional (default=0.99)
        The asymmetry parameter to use for the high-energy side of the dataset.
        This parameter controls the degree of smoothing and should be between 0 and 1.
        Smaller values lead to smoother results.
    plot_results: bool, optional (default=False)
        If True, a plot will be shown with the original and smoothed datasets.

    Returns:
    --------
    A dictionary with two keys:
    - 'min_smoothed': an xr.Dataset object witch is generated by whittaker_eilers_smoother() "asym_min" 
    - 'max_smoothed': -----//------ "asym_max"
    """
    
    points_distats = abs(test_data.eV.data[0]-test_data.eV.data[1])/2 # halfe of the distance between eV (eV step). Just for exclution of inflaction poitn from dataset
    left = test_data.sel(eV = slice(inflection_point-points_distats))
    rigth = test_data.sel(eV = slice(inflection_point+points_distats, max(test_data.eV)))
    left_down_smoth = whittaker_eilers_smoother(left,
                                  asym = asym_min)
    right_down_smoth = whittaker_eilers_smoother(rigth,
                                  asym = asym_max)

    left_up_smoth = whittaker_eilers_smoother(left,
                                  asym = asym_max)
    right_up_smoth = whittaker_eilers_smoother(rigth,
                                  asym = asym_min)
    sample = xr.concat([left_down_smoth[0],
    right_down_smoth[0]], dim = 'eV')
    sample_down_smooth = whittaker_eilers_smoother(sample)
    sample = xr.concat([left_up_smoth[0],
    right_up_smoth[0]], dim = 'eV')
    sample_up_smooth = whittaker_eilers_smoother(sample)
    if plot_results:
        plt.cla()
        #test_data.plot(marker = 'o')
        test_data.plot(marker = '.', ls = '', label ='data', zorder = 100)

        sample_up_smooth[0].plot(marker = '.', label = 'min weighting')
        sample_down_smooth[0].plot(marker = '.', label = 'max weighting')

        plt.legend()
        plt.show()
    return {'min_smoothed': sample_up_smooth, 'max_smoothed': sample_down_smooth}

def find_intersec(tangente_roi, bkg_roi):
    """
    Finde the intersection point of two lines, witch are fitted on two two different regions (tangente_roi, bkg_roi)

    Parameters:
        tangente_roi  [array or list]: points where line fit for tangenten_roi shuld be done
        bkg_roi       [array or list]: same for background

    Returns: 
        dict: {'intersection_eV':        -> Intersection point  
                'intersection_eV_err':   -> Error calculatet just from errors of two fits
            }
    """
    
    tangente_fit = LinearModel().guess_fit(tangente_roi,
                                          weights = 1./np.sqrt(tangente_roi.data))
    bkg_roi_fit = LinearModel().guess_fit(bkg_roi,
                                          weights = 1./np.sqrt(bkg_roi.data)) 
    #### Translate lmfit to dictionary, because lmfit output sucks
    tangenten_results = {}
    for i in tangente_fit.params.values():
        tangenten_results[i.name]={'value':i.value, 'stderr':i.stderr}

    bkg_results = {}
    for i in bkg_roi_fit.params.values():
        bkg_results[i.name]={'value':i.value, 'stderr':i.stderr}

    #rename constanst just for convinien and clarity
    b1 = bkg_results['intercept']['value']
    b2 = tangenten_results['intercept']['value']
    a1 = bkg_results['slope']['value']
    a2 = tangenten_results['slope']['value']
    delta_b1 = bkg_results['intercept']['stderr']
    delta_b2 = tangenten_results['intercept']['stderr']
    delta_a1 = bkg_results['slope']['stderr']
    delta_a2 = tangenten_results['slope']['stderr']
    #print(delta_a1,delta_a2, delta_b1, delta_b2)

    ## calcluate intersection of lines and there errors    
    #x_intersection = (bkg_roi_fit.values['intercept'] - tangente_fit.values['intercept']) / (tangente_fit.values['slope'] - bkg_roi_fit.values['slope'])
    #x_intersection = (bkg_results['intercept']['value'] - tangenten_results['intercept']['value']) / (tangenten_results['slope']['value'] - bkg_results['slope']['value'] ) 
    x_intersection = (b1-b2)/(a2-a1)

    ### Calculatin ERROR of interection of two lines. Error Propagation
    one = delta_b1 / (a2 - a1)
    two = delta_b2 / (a1 - a2)
    tree = ((b2 - b1) * delta_a2) / (a2 - a1)**2.
    four = ((b1 - b2) * delta_a1) / (a2 - a1)**2.
    x_intersection_err = np.sqrt(one**2. + two**2. + tree**2. + four**2.)
    return {'intersection_eV': x_intersection, 'intersection_eV_err': x_intersection_err, 'tangente_roi_fit' : tangente_fit, 'bkg_roi_fit' : bkg_roi_fit}
