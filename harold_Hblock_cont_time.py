'''
D.Glover PNNL Power Electronics Team Ph.D. intern, Summer 2022
HWPV dynamic modeling continuous time LTI conversion
HWPV H-Block Conversion H(z) ---> H(s) from "_*_fhf_poly.json
Harold Control Systems Toolbox 'Undiscretize' using 'Forward Eulers' approximation method
This script writes model coeffs to a new json folder H1s OR the existing fhf_poly.json model - uncomment below to use
'''


import numpy as np
import harold as har
print(har.__version__)
import control
import json


''' ** SET GLOBALS HERE ** '''
# local path to models folder
models_path = 'C:/Users/glov299/PycharmProjects/pecblocks/examples/hwpv/models/'
# time step
dt = 0.001
# adjust json models for use case read/write
fhf_poly_model = 'pv1_fhf_poly.json' # fhf read in
h1s_json_new = 'pv1_H1s.json' # write to new json


# read in Json FHF from models as dict
def readInFHFModelJson():
    with open(models_path + fhf_poly_model,'rb') as fhf_read:
        fhf_dict = json.load(fhf_read)
        print(fhf_dict)
        fhf_read.close()
    return fhf_dict # contains json fhf file from 'models'


# Read in H1 coeffs model from FHF json file with number inputs and outputs
def H1CoeffsToDictJson(mydict):
    h1 = {}
    h1_coeffs = {}
    for key,val in mydict.items():
        if key == 'H1':
            h1.update({key:val})
            for subkey, subval in val.items():
                if subkey =='n_in':
                    print("number H-Block inputs =", subval)
                    num_ins = subval
                elif subkey =='n_out':
                    print("number H-block outputs =", subval)
                    num_outs = subval
                elif subkey[0] == 'a':
                    subval.insert(0,1) # add one for z^8
                    h1_coeffs.update({subkey:subval})
                elif subkey[0] == 'b':
                    h1_coeffs.update({subkey:subval})
                else: pass
    return h1_coeffs, num_ins, num_outs, h1 # return coeffs dict, i/o model values


# Coeffs to NumPy arrays for control
def HCoeffsToNumpy(hdict):
    aarr = []
    barr = []
    aarr2 = []
    barr2 = []
    for param,value in hdict.items():
        if param[0] == 'a':
            aarr.append(value)
            if int(param[4]) == num_outputs - 1: # index for last output in row
                aarr2.append(aarr)
                aarr = []
        if param[0] == 'b':
            barr.append(value)
            if int(param[4]) == num_outputs - 1:
                barr2.append(barr)
                barr = []
    a_array = np.array(aarr2)
    b_array = np.array(barr2)
    return a_array, b_array


# convert Hz_sys_all to dictionary
def HzToDict(Hz_all):
    Hz_sys_all_dictionary = {}
    for i in range(0, num_inputs): # inputs
        for j in range(0, num_outputs): # outputs
            Hz_sys_all_dictionary.update({(i, j): Hz_all[i, j]})
    return Hz_sys_all_dictionary


def toHarold(a,b):
    mydict = {}
    for input in range(0,num_inputs):
        for output in range(0,num_outputs):
            H_z_har = har.Transfer(b[input][output],a[input][output],dt=dt)
            mydict.update({(input,output): H_z_har})
    return mydict # dict of all discrete time tf's in harold systems format {i/o : z-domain tf}


# convert cttf's from Harold back to Controls package
def haroldToControlsContTF(Hs_dict_harold):
    controls_Hs_dict = {}
    for method,Hs in Hs_dict_harold.items():
        numHs = np.array(Hs.num).ravel().tolist()
        denHs = np.array(Hs.den).ravel().tolist()
        Hs = control.TransferFunction(numHs, denHs)
        controls_Hs_dict.update({method: Hs})
    return controls_Hs_dict


# loop through dict and check poles for stability, return dict of stable cttfs ONLY
def checkPoles(Hs_dict):
    stable_dict = {}
    for method,cttf in Hs_dict.items():
        poles = control.pole(cttf)
        real_poles = np.real(poles)
        if np.all(real_poles < 0):
            stable_dict.update({method: cttf})
            # print(method + ' ' + 'method H(s) is stable')
        else:
            # print(method + ' ' + 'method H(s) is unstable')
            pass
    return stable_dict


# convert H(z)_sys_all_har to H(s) for each i/o using Forward Eulers
def convertHzToHsAll(Hz_dict_harold):
    Hs_dict_harold = {}
    for io,tf in Hz_dict_harold.items():
        Hs = har.undiscretize(tf,method='forward euler',prewarp_at=499.99,q='none') # prewarp only for 'tustin' method
        Hs_dict_harold.update({io:Hs})
    return Hs_dict_harold


# Sample Continuous H(s) tf's for equivalent Z-domain tf to compare with original
def sampleHs(Hs_sys, sample_freq):
    Hz_dict = {}
    for key,val in Hs_sys.items():
        Hz_new = control.sample_system(val,sample_freq, 'euler')
        Hz_dict.update({key:Hz_new})
    return Hz_dict


# convert transfer function --> numpy arrays list
def getHsCoeffs (Hs_dict):
    denom = []
    num = []
    for key,value in Hs_dict.items():
        val = control.tfdata(value)
        num.append(val[0]) # b coeffs
        denom.append(val[1]) # a coeffs
    return num,denom


# convert num/denom coeffs into single dimension arrays (lists)
def stripHsCoeffsToList(mylist):
    coeffs_list = []
    flat_list = [item for sublist in mylist for item in sublist]
    new_flat_list = [item for sublist in flat_list for item in sublist]
    for newitem in new_flat_list:
        if newitem[0] == 1: # remove added '1' from denominator (a) coeffs
            coeffs_list.append(newitem[1:].tolist())
        else:
            coeffs_list.append(newitem.tolist())
    return coeffs_list


# make dict from two lists of a,b coeffs
def makeH1sDict(lista, listb, h1dict_copy):
    ordered_coeffs = []
    h1s_new = {}
    h1s_final = {}
    for itema, itemb in zip(lista, listb):
        ordered_coeffs.append(itema)
        ordered_coeffs.append(itemb)
    for key, value in h1dict_copy.items():
        if key == 'H1':
            index = 0
            for subkey, subval in value.items():
                if subkey[0] == 'a' or subkey[0] == 'b':
                    h1s_new.update({subkey : ordered_coeffs[index]})
                    index += 1
                else:
                    h1s_new.update({subkey : subval})
    h1s_final.update({'H1s': h1s_new}) # add 'H1s' key name to front of dict
    return h1s_final


# write final H1s dict to json file
def writeH1sNewJson(mydict):
    with open(models_path + h1s_json_new, 'w') as h1s_write:
        json.dump(mydict, h1s_write, indent=2)
        h1s_write.close()


# write H1(s) to fhf_poly.json
def addH1sToFHFPolyJson(h1s, fhf_json):
    fhf_json.update(h1s)
    with open(models_path + 'pv1_fhf_new_poly.json', 'w') as h1s_write:
        json.dump(fhf_json, h1s_write, indent=2)
        h1s_write.close()


if __name__ == '__main__':

    # pull in fhf json
    json_fhf = readInFHFModelJson()
    h1_coeffs_dict, num_inputs, num_outputs, h1_copy = H1CoeffsToDictJson(json_fhf)
    a_coeffs, b_coeffs = HCoeffsToNumpy(h1_coeffs_dict)

    # all tf's in controls lib form
    Hz_sys_all = control.TransferFunction(b_coeffs, a_coeffs, dt)
    # dict holds all original H(z) transfer functions
    Hz_sys_all_dict = HzToDict(Hz_sys_all)

    # get each tf for each i/o to Harold format
    Hz_sys_all_har = toHarold(a_coeffs, b_coeffs)

    # conv all H(z) to H(s) for each i/o and check stability using forward euler/difference approx
    Hs_sys_all_har = convertHzToHsAll(Hz_sys_all_har)
    Hs_sys_all_ctrls = haroldToControlsContTF(Hs_sys_all_har)
    Hs_sys_all_stable = checkPoles(Hs_sys_all_ctrls)

    # create discrete time tf H(z) to compare with original from pv1_import.py
    # Hz_one_msec = sampleHs(Hs_sys_all_stable, dt)  # should be equivalent to original H(z)

    # strip H(s) transfer function coeffs to arrays for rewrite
    b_arrays, a_arrays = getHsCoeffs(Hs_sys_all_stable)
    b_coeffs_Hs = stripHsCoeffsToList(b_arrays)
    a_coeffs_Hs = stripHsCoeffsToList(a_arrays)
    h1s_new_dict = makeH1sDict(a_coeffs_Hs, b_coeffs_Hs, h1_copy)

    '''Uncomment below to write new Hs file dict.json OR append H1s dict to existing fhf_poly.json'''
    # write H1s dict to json, uncomment to write new json file
    # writeH1sNewJson(h1s_new_dict)

    # addH1sToFHFPolyJson(h1s_new_dict, json_fhf)
    
print("Conversion complete")
