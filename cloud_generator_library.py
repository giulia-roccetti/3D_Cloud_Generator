import numpy as np
import random
import scipy as sp

def correlation_param(z_level, decorr_length):
    corr = []
    for iz in range(0,len(z_level)-1):
        corr.append(np.exp(-np.abs(z_level[iz+1]-z_level[iz]) / decorr_length ))
    corr.append(np.exp(-np.abs(z_level[-1]) / decorr_length))
    return corr

def gamma_function(mean, sigma):
    a = pow(mean/sigma,2)
    b = a / mean
    x_gamma = np.linspace(0,10,1000)
    y_gamma = sp.stats.gamma.cdf(x_gamma, a, scale=1/b)
    return x_gamma, y_gamma

def get_cumulative_cc(overlap_method, cc, alpha):

    # Initialization of combined cloud cover of adjiacent pair of layers (p) and the cumulative cloud cover (cumulative_cc)
    p = np.zeros(len(cc))
    cumulative_cc = np.zeros(len(cc))

    # First element of cumulative_cc is equal to the cloud cover at TOA
    cumulative_cc[0] = cc[0]

    # Cycle over different layers i, from TOA - 1
    for i in range(1, len(cc)):

        # Select overlap method and relative combined cloud cover (p) equation
        if (overlap_method == 'max_ran'):
            p[i] = max(cc[i], cc[i-1])
        else: 
            p[i] = alpha[i-1] * max(cc[i], cc[i-1]) + (1 - alpha[i-1]) * (cc[i-1] + cc[i] - cc[i] * cc[i-1])
        
        # General equation for cumulative_cc as function of p
        if(cc[i-1] == 1):
            cumulative_cc[i] = cumulative_cc[i-1]
        else:
            cumulative_cc[i] = 1 - ((1 - cumulative_cc[i-1]) * (1 - p[i]) / (1 - cc[i-1]))
    
    # Last element of cumulative_cc is the total cloud cover calculated with the specified method
    return cumulative_cc, p

def fill_cloudy_layers(f_zoom, c_function, p_function, cc, lwc, iwc, lwc_corr):

    # Initialization of cloud cover, lwc and iwc matrices
    cc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))
    lwc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))
    iwc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))

    if(c_function[-1] > 0.1):
        # extract x and y from a Gamma function with mean of 1 and fractional sigma 0.75, to be used to extract lwc and iwc values
        x_gamma, y_gamma = gamma_function(1, 0.75)

        # Indices of subcolumns to be filled, calculated as the total cloud cover * number of subcolumns
        # We sample the range from 0 to number of subcolumns and select the correct number of random indices to be filled
        j_cloudy = random.sample(range(f_zoom * f_zoom), int(np.round(c_function[-1] * f_zoom * f_zoom)))

        # Cycle over subcolumns
        for j in j_cloudy:

            # First random number
            R0 = np.random.rand()

            # We define the top and bottom cloudy layers
            i_top = np.around(np.where((c_function / c_function[-1]) - R0 > 0, (c_function / c_function[-1]) - R0, np.inf), 5).argmin()
            i_base = np.max(np.nonzero(cc))

            # Initialization of the first cloudy layer
            cc_matrix[j, i_top] = 1

            # We place the first lwc and iwc at the first cloudy layer from TOA
            # The value of the lwc and iwc are extracted from a Gamma distribution with mean value 1 and fractional standard deviation of 0.75
            # The value of lwc and iwc is than scaled with the in-cloud lwc and iwc (lwc/cc and iwc/cc)
            lwc_matrix[j, i_top] = x_gamma[np.abs(y_gamma - np.random.rand()).argmin()] * lwc[i_top] / cc[i_top]
            iwc_matrix[j, i_top] = x_gamma[np.abs(y_gamma - np.random.rand()).argmin()] * iwc[i_top] / cc[i_top]

            # Cycle over cloudy layers
            for i in range(i_top, i_base):

                ## Placement of clouds 
                # Second random number
                R1 = np.random.rand()

                # If previous layer is cloudy, a cloud is present in the next layer is R1 is smaller than this quantity (Hogan and Bozzo, 2018, Eq. 4)
                if (cc_matrix[j, i] == 1): 
                    if (R1 < (cc[i] + cc[i+1] - p_function[i+1]) / cc[i]): 
                        cc_matrix[j, i+1] = 1
                
                # If previous layer is non-cloudy, the next layer is cloudy if R1 is smaller than this quantity (Hogan and Bozzo, 2018, Eq. 5)
                else:
                    if (R1 < (p_function[i+1] - cc[i] - c_function[i+1] + c_function[i]) / (c_function[i] - cc[i])): 
                        cc_matrix[j, i+1] = 1 


                #### Cloud heterogeneity part
                # Third and fourth random number
                R2 = np.random.rand()
                R3 = np.random.rand()

                # If the next layer is cloudy, we fill it with lwc and iwc
                if (cc_matrix[j, i+1] == 1): 

                    # If R2 is smaller than the linear correlation between cloud condensate cumulative frequencies (lwc_corr) we use the lwc of the previous layer
                    if ((R2 <= lwc_corr[i]) & (cc_matrix[j,i] == 1)):
                        lwc_matrix[j,i+1] = lwc_matrix[j,i] 

                    # Otherwise, we extract another lwc and iwc from the gamma distribtion
                    else:
                        lwc_matrix[j,i+1] = x_gamma[np.abs(y_gamma - np.random.rand()).argmin()] * lwc[i+1] / cc[i+1]

                    # Same procedure for iwc, based on R3
                    if ((R3 <= lwc_corr[i]) & (cc_matrix[j,i] == 1)):
                        iwc_matrix[j,i+1] = iwc_matrix[j,i] 
                    else:
                        iwc_matrix[j,i+1] = x_gamma[np.abs(y_gamma - np.random.rand()).argmin()] * iwc[i+1] / cc[i+1]

        # Now we need to rescale the lwc and iwc values to average the ERA5 over a layer
        lwc_matrix_old = np.copy(lwc_matrix)
        iwc_matrix_old = np.copy(iwc_matrix)
        j_c, i_c = np.where(cc_matrix != 0)
        for k in range(len(j_c)):
            if (np.sum(lwc_matrix_old[:,i_c[k]]) !=0):
                lwc_matrix[j_c[k],i_c[k]] *= (lwc[i_c[k]] / cc[i_c[k]]) / (np.sum(lwc_matrix_old[:,i_c[k]]) / np.sum(cc_matrix[:,i_c[k]]))
            if (np.sum(iwc_matrix_old[:,i_c[k]]) !=0):
                iwc_matrix[j_c[k],i_c[k]] *= (iwc[i_c[k]] / cc[i_c[k]]) / (np.sum(iwc_matrix_old[:,i_c[k]]) / np.sum(cc_matrix[:,i_c[k]]))

        # Conservation of the optical thickness of clouds
        # If there is lwc or iwc in a layer, but no clouds were added in the deterministic part, we put an additional cloud with all the in-cloud lwc and iwc
        # This additional cloud is always placed in the first subcolumn
        for i in range(len(cc)):
            if((lwc[i] > 0) & (np.sum(cc_matrix[:,i]) == 0) & (cc[i] > 0)):
                cc_matrix[0,i] = 1
                lwc_matrix[0,i] = lwc[i] / cc[i]
                
            if((iwc[i] > 0) & (np.sum(cc_matrix[:,i]) == 0) & (cc[i] > 0)):
                cc_matrix[0,i] = 1
                iwc_matrix[0,i] = iwc[i] / cc[i]

        # Last condition to put a cutoff of cloudy pixels. If the total cloud cover of a column is less than 10%, we remove the cloud
        if(c_function[-1] < 0.10):
            cc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))
            lwc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))
            iwc_matrix = np.zeros((f_zoom * f_zoom, len(cc)))

    return cc_matrix, lwc_matrix, iwc_matrix
