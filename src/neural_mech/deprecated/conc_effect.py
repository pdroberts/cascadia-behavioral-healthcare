import numpy as np
import pandas as pd

def MichaelisMenten(K_i, C): # K_i: affinity ligand, C: concentration ligand 
    act = C/(K_i+C)
    return act

def CompetitiveBinding(K_A, C_A, K_B, C_B): # K_A: affinity ligand A, C_A: concentration ligand A 
    a = K_A + K_B + C_A + C_B - 1
    b = K_B*(C_A - 1) + K_A*(C_B - 1) + K_A*K_B
    c = -K_A*K_B
    theta = np.arccos( (-2*a**3+9*a*b-27*c) / (2*np.sqrt((a**2-3*b)**3)) )
    return C_A*(2*np.sqrt(a**2-3*b)*np.cos(theta/3)-a)/(3*K_A+(2*np.sqrt(a**2-3*b)*np.cos(theta/3)-a))


def AveConcentration(drugParams, dose, doseInterval):  # dose (mg), doseInterval (hr)
    F = drugParams[0]*100   # bioavailability (fraction)
    CL = drugParams[1]  # clearance (L/hr)
    C_ave_plasma = F*dose/(CL*doseInterval)
    C_ave_brain = C_ave_plasma/drugParams[2]  # blood/brain ratio, C_ave_brain (mg/L)
    C_ave = 1000*C_ave_brain/drugParams[3]         # mole wt (g/mol)
    return C_ave*drugParams[4]    # pk_param

def Dose2Conc(szDrugs, drugParams): # returns a copy of the drug dataframe with concentratiosn in place of dose
    concDrug = szDrugs.copy()
    for drug in szDrugs.keys():
        if drug not in drugParams.keys(): d = 'unknown'
        else: d = drug
        concDrug[drug] = AveConcentration(drugParams[d], szDrugs[drug], 24)
    return concDrug

def Lamotrigine_Na(C_L): # Na-block,  C_L: Lamotrigine concentration 
	C_L = C_L/100.  #-!!! FACTOR of 1/100 to reduce effect for calculated brain concentration !!!
	K_C = 513 # (uM)
	n = 0.9
	d_INa = 1 - C_L**n/(C_L + K_C)**n # %-change in Na current [Xie95]
	return d_INa

def Lamotrigine_Ih(C_L): # Rate change cause by I_h activation shift, C_L: Lamotrigine concentration (uM) 
	C_L = C_L/100.  #-!!! FACTOR of 1/100 to reduce effect for calculated brain concentration !!!
	d_re = 1 - 0.004 * C_L/100.  # fractional-change in pyramidal cell firing rate [Poolos02, Fig6a] 
	return d_re

def Lamotrigine_glu(C_L): # Na-block,  C_L: Lamotrigine concentration 
	C_L = C_L/100.  #-!!! FACTOR of 1/100 to reduce effect for calculated brain concentration !!!
	d_we = 1 - 0.0045 * C_L  # fractional-change in glu release [Wang01f] 
	return d_we

# def Lorazepam_GABA_occ_Miller78b(C_L): # GABA agonist,  C_L: Lorazepam concentration 
# 	A = 1.6555 
# 	B = 3296 # (uM)
# 	d_gaba = C_L**A / (C_L**A + B) # %-change in GABA current [Miller87b]
# 	return d_gaba *0.4 #Fudge to modulate relative effect of benzos
# 
# def Clonazepam_GABA_occ_Miller78b(C_L): # GABA agonist,  C_L: Clonazepam concentration 
# 	A = 1.4328 
# 	B = 230 # (uM)
# 	d_gaba = C_L**A / (C_L**A + B) # %-change in GABA current [Miller87b]
# 	return d_gaba *0.4 #Fudge to modulate relative effect of benzos

def Lorazepam_GABA_occ(C_L): # GABA agonist,  C_L: Lorazepam concentration 
	K_i = 2.3  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
	d_gaba = MichaelisMenten(K_i, C_L) # %-change in GABA current 
	return d_gaba *0.4 #Fudge to modulate relative effect of benzos

def Clonazepam_GABA_occ(C_L): # GABA agonist,  C_L: Clonazepam concentration 
	K_i = 0.87  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
	d_gaba = MichaelisMenten(K_i, C_L) # %-change in GABA current 
	return d_gaba *0.4 #Fudge to modulate relative effect of benzos

def Diazepam_GABA_occ(C_L): # GABA agonist,  C_L: Diazepam concentration 
	K_i = 7.4  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
	d_gaba = MichaelisMenten(K_i, C_L) # %-change in GABA current 
	return d_gaba *0.4 #Fudge to modulate relative effect of benzos

def Zolpidem_GABA_occ(C_L): # GABA agonist,  C_L: Zolpidem concentration 
	K_i = 56.3  # [Sancar07] 
	d_gaba = MichaelisMenten(K_i, C_L) # %-change in GABA current 
	return d_gaba *0.4 #Fudge to modulate relative effect of benzos

def receptor_occupation(concDrug, receptor, Ki_receptorDrug, concEndog):
	conc = concDrug
	receptorActivation = {} 
	receptorAct = {}
	if receptor=='D2':
		# -- D2 --
		actSet = []
		for d in conc.keys():
			K_B = Ki_receptorDrug['DOPAMINE']['D2']
			C_B = (concEndog['dopamine_tonic'] + concEndog['dopamine_burst'])/2
			K_A = Ki_receptorDrug[d]['D2']
			C_A = conc[d]
		receptorAct['D2'] = CompetitiveBinding(K_A, C_A, K_B, C_B)
	return CompetitiveBinding(K_A, C_A, K_B, C_B).values


def receptor_binding(concDrug, Ki_receptorDrug, concEndog):
	con = concDrug.loc[0,:].copy()
	con.name = 'control'
	con.loc[:] = 0
	concDrug = concDrug.append(con)
	receptorActivation = {} 
	for subject in concDrug.index:
		conc = concDrug.loc[subject]#[concDrug.loc['FP_MT']!=0]
		receptorAct = {}
		# -- D1 --
		actSet = []
		for d in conc.keys():
			K_A = Ki_receptorDrug['DOPAMINE']['D1']
			C_A = (concEndog['dopamine_tonic'] + concEndog['dopamine_burst'])/2
			K_B = Ki_receptorDrug[d]['D1']
			C_B = conc[d]
			actSet.append( CompetitiveBinding(K_A, C_A, K_B, C_B) )
			act = np.mean(actSet)
		receptorAct['D1'] = act
		# -- D2 --
		actSet = []
		for d in conc.keys():
			K_A = Ki_receptorDrug['DOPAMINE']['D2']
			C_A = (concEndog['dopamine_tonic'] + concEndog['dopamine_burst'])/2
			K_B = Ki_receptorDrug[d]['D2']
			C_B = conc[d]
			actSet.append( CompetitiveBinding(K_A, C_A, K_B, C_B) )
			act = np.mean(actSet)
		receptorAct['D2'] = act
		# -- 5-HT1A --
		actSet = []
		for d in conc.keys():
			K_A = Ki_receptorDrug['5-Hydroxy Tryptamine']['5-HT1A']
			C_A = concEndog['serotonin']
			K_B = Ki_receptorDrug[d]['5-HT1A']
			C_B = conc[d]
			actSet.append( CompetitiveBinding(K_A, C_A, K_B, C_B) )
			act = np.mean(actSet)
		receptorAct['5-HT1A'] = act
		# -- 5-HT2A --
		actSet = []
		for d in conc.keys():
			K_A = Ki_receptorDrug['5-Hydroxy Tryptamine']['5-HT2A']
			C_A = concEndog['serotonin']
			K_B = Ki_receptorDrug[d]['5-HT2A']
			C_B = conc[d]
			actSet.append( CompetitiveBinding(K_A, C_A, K_B, C_B) )
			act = np.mean(actSet)
		receptorAct['5-HT2A'] = act
		# -- M1 --
		actSet = []
		for d in conc.keys():
			K_A = Ki_receptorDrug['Acetylcholine']['M1']
			C_A = concEndog['Acetylcholine']
			K_B = Ki_receptorDrug[d]['M1']
			C_B = conc[d]
			actSet.append( CompetitiveBinding(K_A, C_A, K_B, C_B) )
			act = np.mean(actSet)
		receptorAct['M1'] = act
		receptorActivation[subject] = receptorAct
	return pd.DataFrame(receptorActivation).T  
	
def conc_moa_effect(concDrug, moa_activation):
    effect = moa_activation.copy()
    if 'GABA' not in effect.keys():
        effect['GABA'] = np.zeros(len(effect))
    for c in concDrug.index:
        if 'Lamotrigine' in concDrug.keys(): 
            effect.loc[c, 'glu'] = Lamotrigine_glu(concDrug.loc[c, 'Lamotrigine'])
            effect.loc[c, 'Na'] = Lamotrigine_Na(concDrug.loc[c, 'Lamotrigine'])
            effect.loc[c, 'Ih'] = Lamotrigine_Ih(concDrug.loc[c, 'Lamotrigine']) # should be (4*dx1/mu1)
        if 'Lorazepam' in concDrug.keys():
            effect.loc[c, 'GABA'] = effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*Lorazepam_GABA_occ(concDrug.loc[c, 'Lorazepam'])
        if 'Diazepam' in concDrug.keys():
            effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*Diazepam_GABA_occ(concDrug.loc[c, 'Diazepam'])
        if 'Clonazepam' in concDrug.keys():
            effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*Clonazepam_GABA_occ(concDrug.loc[c, 'Clonazepam'])
        if 'Zolpidem' in concDrug.keys():
            effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*Zolpidem_GABA_occ(concDrug.loc[c, 'Zolpidem'])
    effect.loc['control', ['glu','Na','Ih']] = 1.
    return effect

def act_to_model_param(activation, modelParams):
    effect = modelParams.copy()
    for c in activation.index[:-1]:   # eliminate last index because it is the 'control'
        r = 'D1'
        effect.loc[c, 'mu1'] = effect.loc[c, 'mu1'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'th0'] = effect.loc[c, 'th0'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'w11'] = effect.loc[c, 'w11'] \
            *(1+(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'w10'] = effect.loc[c, 'w10'] \
            *(1+(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'w01'] = effect.loc[c, 'w01'] \
            *(1+(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        r = 'D2'
        effect.loc[c, 'mu1'] = effect.loc[c, 'mu1'] \
            *(1+(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'w11'] = effect.loc[c, 'w11'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        effect.loc[c, 'w01'] = effect.loc[c, 'w01'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        r = '5-HT1A'
        effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
            *(1+(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        r = '5-HT2A'
        effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        r = 'M1'
        effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
            *(1-(activation.loc[c,r] - activation.loc['control',r])/activation.loc['control',r])
        r = 'GABA'
        effect.loc[c, 'w01'] = effect.loc[c, 'w01']*(1+activation.loc[c,r])
        effect.loc[c, 'w00'] = effect.loc[c, 'w00']*(1+activation.loc[c,r])
        r = 'glu'
        effect.loc[c, 'w11'] = effect.loc[c, 'w11']*activation.loc[c,r]
        effect.loc[c, 'w10'] = effect.loc[c, 'w10']*activation.loc[c,r]
        r = 'Na'
        effect.loc[c, 'th1'] = effect.loc[c, 'th1']/activation.loc[c,r]
        r = 'Ih'
        effect.loc[c, 'th1'] = effect.loc[c, 'th1']/activation.loc[c,r]
    return effect
    
    
def ConcMetabEffect(activation, modelParams):
	effect = modelParams.copy()
	for c in activation.index[:-1]:   # eliminate last index because it is the 'control'
		r = 'D1'
		effect.loc[c, 'mu1'] = effect.loc[c, 'mu1'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'th0'] = effect.loc[c, 'th0'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'w11'] = effect.loc[c, 'w11'] \
			*(1+(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'w10'] = effect.loc[c, 'w10'] \
			*(1+(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'w01'] = effect.loc[c, 'w01'] \
			*(1+(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		r = 'D2'
		effect.loc[c, 'mu1'] = effect.loc[c, 'mu1'] \
			*(1+(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'w11'] = effect.loc[c, 'w11'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		effect.loc[c, 'w01'] = effect.loc[c, 'w01'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		r = '5-HT1A'
		effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
			*(1+(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		r = '5-HT2A'
		effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
		r = 'M1'
		effect.loc[c, 'th1'] = effect.loc[c, 'th1'] \
			*(1-(activation.loc['control',r]-activation.loc[c,r])/activation.loc['control',r])
	return effect

def ConcIonoEffect(concDrug, modelParams):
	effect = modelParams.copy()
	for c in concDrug.index:
		if 'Lamotrigine' in concDrug.keys(): 
			effect.loc[c, 'w10'] = effect.loc[c, 'w10']*Lamotrigine_glu(concDrug.loc[c, 'Lamotrigine'])
			effect.loc[c, 'w11'] = effect.loc[c, 'w11']*Lamotrigine_glu(concDrug.loc[c, 'Lamotrigine'])
			effect.loc[c, 'th1'] = effect.loc[c, 'th1']*Lamotrigine_Na(concDrug.loc[c, 'Lamotrigine'])
			effect.loc[c, 'th1'] = effect.loc[c, 'th1']*Lamotrigine_Ih(concDrug.loc[c, 'Lamotrigine']) # should be (4*dx1/mu1)
		if 'Lorazepam' in concDrug.keys():
			effect.loc[c, 'w01'] = effect.loc[c, 'w01']*(1+Lorazepam_GABA_occ(concDrug.loc[c, 'Lorazepam']))
			effect.loc[c, 'w00'] = effect.loc[c, 'w00']*(1+Lorazepam_GABA_occ(concDrug.loc[c, 'Lorazepam']))
		if 'Diazepam' in concDrug.keys():
			effect.loc[c, 'w01'] = effect.loc[c, 'w01']*(1+Diazepam_GABA_occ(concDrug.loc[c, 'Diazepam']))
			effect.loc[c, 'w00'] = effect.loc[c, 'w00']*(1+Diazepam_GABA_occ(concDrug.loc[c, 'Diazepam']))
		if 'Clonazepam' in concDrug.keys():
			effect.loc[c, 'w01'] = effect.loc[c, 'w01']*(1+Clonazepam_GABA_occ(concDrug.loc[c, 'Clonazepam']))
			effect.loc[c, 'w00'] = effect.loc[c, 'w00']*(1+Clonazepam_GABA_occ(concDrug.loc[c, 'Clonazepam']))
	return effect

def dose_to_model_param(szDrugs, drugParams, Ki_receptDrug, endogNT):
    # Ki_receptDrug, endogNT defined externally
    concDrug = Dose2Conc(szDrugs[szDrugs.keys()], drugParams).fillna(0)
    act = conc_moa_effect(concDrug, receptor_binding(concDrug, Ki_receptDrug, endogNT))
    modelParams = pd.DataFrame(1, index=szDrugs.index, \
                           columns=['mu0', 'mu1', 'th0', 'th1', \
                                    'w00',  'w01',  'w10',  'w11', ])
    return ConcIonoEffect(concDrug, ConcMetabEffect(act, modelParams))

