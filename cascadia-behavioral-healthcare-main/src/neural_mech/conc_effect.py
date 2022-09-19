#!/usr/bin/python
# conc_effect.py by Patrick D Roberts (2022)
# Calculate concentrations and effects on Wilson-Cowan model parameters from dose
# import neural_mech.dose_to_model as dose2model
# d2m = dose2model.Dose_Conversion()
import numpy as np
import pandas as pd

def seperate_clinical_pre_post(clinical_data):
    pre_header = ['patient','TotalPre'] \
                    + [s for s in clinical_data.columns if 'pre_' in s]
    pre_clinical = clinical_data[pre_header].copy()
    pre_clinical.columns = ['patient','pre_total_bf'] \
                    + [str(col)[4:] for col in list(pre_clinical.columns)[2:]]

    post_header = ['patient','TotalPost'] \
                    + [s for s in clinical_data.columns if 'post_' in s]
    post_clinical = clinical_data[post_header].copy()
    post_clinical.columns = ['patient','post_total_bf'] \
                    + [str(col)[5:] for col in list(post_clinical.columns)[2:]]
    return pre_clinical, post_clinical

class Dose_Conversion:    
    """Class for handling medication data to model.
    :param p: input parameter
    :type p: int
    """
    def __init__(self, clinical_data=None):
        self.clinical_data = clinical_data
         
         
    def load_pharmacokinetic_info(self, pharma_pk):
        # Load pharmacokinetic information on drugs
        self.pharma_pk = pharma_pk

    def load_drug_affinities(self, pk_data):
        # Read drug affinity data from PDSP 𝐾𝑖 database 
        # http://pdsp.med.unc.edu/downloadKi.html.
        # Put the average 𝐾𝑖 values for choice antipsychotics and drugs in a DataFrame.
        self.pk_data = pk_data
        drugs = list(self.clinical_data.keys())
        drugs.sort()
        nt = ['DOPAMINE', '5-Hydroxy Tryptamine', 'Acetylcholine']
        drugs = nt + drugs

        receptors = ['DOPAMINE D1','DOPAMINE D2','5-HT1A','5-HT2A', 'Cholinergic, muscarinic M1']
        affinities = pd.DataFrame()
        for r in receptors:
            aff = []
            for d in drugs:
                affList = self.pk_data[(self.pk_data[4].str.upper()==d.upper()) 
                                       & (self.pk_data[1]==r) 
                                       & (self.pk_data[9]=='HUMAN')][12]
                if d == 'DOPAMINE' and r == 'DOPAMINE D1': 
                    aff.append(affList.min())
                elif d == 'DOPAMINE' and r == 'DOPAMINE D2': 
                    aff.append(affList[(affList<1000) & (affList>100)].mean())  # Eliminate extreme low and high affinity receptors subunits
                elif d == '5-Hydroxy Tryptamine' and r == '5-HT1A': 
                    aff.append( affList[affList<100].mean() )       # Eliminate receptors with high affinity subunits
                elif d == '5-Hydroxy Tryptamine' and r == '5-HT2A': 
                    aff.append(affList[affList<100].mean())         # Eliminate receptors with high affinity subunits
                elif d == 'Acetylcholine' and r == 'Cholinergic, muscarinic M1': 
                    aff.append(affList.min())
                elif len(affList)>0:
                    aff.append(affList.min())
                else:
                    aff.append(self.pk_data[(self.pk_data[4].str.upper()==d.upper()) 
                                            & (self.pk_data[1]==r)][12].min())
            affinities = affinities.append(pd.Series(aff), ignore_index=True)
        affinities.columns = drugs
        affinities['receptors'] = ['D1', 'D2', '5-HT1A', '5-HT2A', 'M1'] #receptors
        self.ki_receptor_drug = affinities.set_index('receptors').fillna(10000)
        self.endo_neurotrans()
        
    def endo_neurotrans(self):
        # Define dictionary of endogenous neurotransmitters
        self.endogNT = {'dopamine_tonic':37, 'dopamine_burst':200, 
           'serotonin':3.9, 'Acetylcholine':10}  # DA [Dreyer10], 5HT [Patterson10], ACh [?] 

    def dose_to_conc(self): # returns a copy of the drug dataframe with concentratiosn in place of dose
        concDrug = self.clinical_data[self.clinical_data.keys()[2:]].copy()
        self.C_ave_plasma = self.clinical_data[self.clinical_data.keys()[2:]].copy()
        for drug in self.clinical_data.keys():
            if drug not in self.pharma_pk.keys(): continue  #d = 'unknown'
            else: d = drug
            concDrug[drug], self.C_ave_plasma[drug] = self.ave_concentration(self.pharma_pk[d], 
                                                                             self.clinical_data[drug], 24)
        return concDrug.fillna(0)
    
    def ave_concentration(self, drug_pharma_pk, dose, doseInterval):  # dose (mg), doseInterval (hr)
        F = drug_pharma_pk[0]   # bioavailability (fraction)
        CL = drug_pharma_pk[1]  # clearance (L/hr)
        C_ave_plasma = F*dose/(CL*doseInterval)*1000    # (ng/mL)
        C_ave_brain = C_ave_plasma/drug_pharma_pk[2]  # blood/brain ratio, C_ave_brain (ng/mL)
        C_ave = C_ave_brain/drug_pharma_pk[3]         # mole wt (g/mol)
        return C_ave*drug_pharma_pk[4], C_ave_plasma    # pk_param
 
    def michaelis_menten(self, K_i, C): # K_i: affinity ligand, C: concentration ligand 
        act = C/(K_i+C)
        return act

    def competitive_binding(self, K_A, C_A, K_B, C_B): # K_A: affinity ligand A, C_A: concentration ligand A 
        a = K_A + K_B + C_A + C_B - 1
        b = K_B*(C_A - 1) + K_A*(C_B - 1) + K_A*K_B
        c = -K_A*K_B
        theta = np.arccos( (-2*a**3+9*a*b-27*c) / (2*np.sqrt((a**2-3*b)**3)) )
        return C_A*(2*np.sqrt(a**2-3*b)*np.cos(theta/3)-a)/(3*K_A+(2*np.sqrt(a**2-3*b)*np.cos(theta/3)-a))


    def lamotrigine_na(self, C_L): # Na-block,  C_L: Lamotrigine concentration 
        K_C = 513 # (uM)
        n = 0.9
        d_INa = 1 - C_L**n/(C_L + K_C)**n  # %-change in Na current [Xie95]
        return d_INa

    def lamotrigine_Ih(self, C_L): # Rate change cause by I_h activation shift, C_L: Lamotrigine concentration (uM) 
        d_re = 1 - 0.004 * C_L  # fractional-change in pyramidal cell firing rate [Poolos02, Fig6a], (4/mu1)?
        return d_re

    def lamotrigine_glu(self, C_L): # Na-block,  C_L: Lamotrigine concentration 
        d_we = 1 - 0.0045 * C_L  # fractional-change in glu release [Wang01f] 
        return d_we

    def lorazepam_gaba_occ(self, C_L): # GABA agonist,  C_L: Lorazepam concentration 
        K_i = 2.3  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
        d_gaba = self.michaelis_menten(K_i, C_L) # %-change in GABA current 
        return d_gaba

    def clonazepam_gaba_occ(self, C_L): # GABA agonist,  C_L: Clonazepam concentration 
        K_i = 0.87  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
        d_gaba = self.michaelis_menten(K_i, C_L) # %-change in GABA current 
        return d_gaba 

    def diazepam_gaba_occ(self, C_L): # GABA agonist,  C_L: Diazepam concentration 
        K_i = 7.4  # [Priest12] (book, see notes_therapeutic_blood_level.txt)
        d_gaba = self.michaelis_menten(K_i, C_L) # %-change in GABA current 
        return d_gaba 

    def zolpidem_gaba_occ(self, C_L): # GABA agonist,  C_L: Zolpidem concentration 
        K_i = 56.3  # [Sancar07] 
        d_gaba = self.michaelis_menten(K_i, C_L) # %-change in GABA current 
        return d_gaba 

    def receptor_binding(self, concDrug):
        con = concDrug.loc[0,:].copy()
        con.name = 'control'
        con.loc[:] = 0
        concDrug = concDrug.append(con)
        receptorActivation = {} 
        for subject in concDrug.index:
            conc = concDrug.loc[subject]
            receptorAct = {}
            # -- D1 --
            actSet = []
            for d in conc.keys():
                K_A = self.ki_receptor_drug['DOPAMINE']['D1']
                C_A = (self.endogNT['dopamine_tonic'] + self.endogNT['dopamine_burst'])/2
                K_B = self.ki_receptor_drug[d]['D1']
                C_B = conc[d]
                actSet.append( self.competitive_binding(K_A, C_A, K_B, C_B) )
                act = np.mean(actSet)
            receptorAct['D1'] = act
            # -- D2 --
            actSet = []
            for d in conc.keys():
                K_A = self.ki_receptor_drug['DOPAMINE']['D2']
                C_A = (self.endogNT['dopamine_tonic'] + self.endogNT['dopamine_burst'])/2
                K_B = self.ki_receptor_drug[d]['D2']
                C_B = conc[d]
                actSet.append( self.competitive_binding(K_A, C_A, K_B, C_B) )
                act = np.mean(actSet)
            receptorAct['D2'] = act
            # -- 5-HT1A --
            actSet = []
            for d in conc.keys():
                K_A = self.ki_receptor_drug['5-Hydroxy Tryptamine']['5-HT1A']
                C_A = self.endogNT['serotonin']
                K_B = self.ki_receptor_drug[d]['5-HT1A']
                C_B = conc[d]
                actSet.append( self.competitive_binding(K_A, C_A, K_B, C_B) )
                act = np.mean(actSet)
            receptorAct['5-HT1A'] = act
            # -- 5-HT2A --
            actSet = []
            for d in conc.keys():
                K_A = self.ki_receptor_drug['5-Hydroxy Tryptamine']['5-HT2A']
                C_A = self.endogNT['serotonin']
                K_B = self.ki_receptor_drug[d]['5-HT2A']
                C_B = conc[d]
                actSet.append( self.competitive_binding(K_A, C_A, K_B, C_B) )
                act = np.mean(actSet)
            receptorAct['5-HT2A'] = act
            # -- M1 --
            actSet = []
            for d in conc.keys():
                K_A = self.ki_receptor_drug['Acetylcholine']['M1']
                C_A = self.endogNT['Acetylcholine']
                K_B = self.ki_receptor_drug[d]['M1']
                C_B = conc[d]
                actSet.append( self.competitive_binding(K_A, C_A, K_B, C_B) )
                act = np.mean(actSet)
            receptorAct['M1'] = act
            receptorActivation[subject] = receptorAct
        return pd.DataFrame(receptorActivation).T  

    def conc_moa_effect(self, concDrug, moa_activation):
        effect = moa_activation.copy()
        if 'GABA' not in effect.keys():
            effect['GABA'] = np.zeros(len(effect))
        for c in concDrug.index:
            if 'Lamotrigine' in concDrug.keys(): 
                p_lam = 0.15
                effect.loc[c, 'glu'] = 1 - p_lam*(1 - self.lamotrigine_glu(concDrug.loc[c, 'Lamotrigine'])) 
                effect.loc[c, 'Na'] = 1 - p_lam*(1 - self.lamotrigine_na(concDrug.loc[c, 'Lamotrigine']))
                effect.loc[c, 'Ih'] = 1 - p_lam*(1 - self.lamotrigine_Ih(concDrug.loc[c, 'Lamotrigine'])) 
            if 'Lorazepam' in concDrug.keys():
                effect.loc[c, 'GABA'] = effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*self.lorazepam_gaba_occ(concDrug.loc[c, 'Lorazepam'])
            if 'Diazepam' in concDrug.keys():
                effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*self.diazepam_gaba_occ(concDrug.loc[c, 'Diazepam'])
            if 'Clonazepam' in concDrug.keys():
                effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*self.clonazepam_gaba_occ(concDrug.loc[c, 'Clonazepam'])
            if 'Zolpidem' in concDrug.keys():
                effect.loc[c, 'GABA'] =  effect.loc[c, 'GABA']+(1-effect.loc[c, 'GABA'])*self.zolpidem_gaba_occ(concDrug.loc[c, 'Zolpidem'])
        effect.loc['control', ['glu','Na','Ih']] = 1.
        return effect

    def act_to_model_param(self, activation, p_wt=1):
        effect = pd.DataFrame(1, index=activation.index, 
                           columns=['mu0','mu1','th0','th1','w00', 'w01', 'w10','w11'])
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
        return self.adj_model_param(effect, p_wt)
    
    def adj_model_param(self, model_param, p_wt=1):
        for index, row in model_param.iterrows():
            for p in ['mu0','mu1','th0','th1','w00', 'w01', 'w10','w11']:
                model_param.loc[index, p] = 1 + p_wt*(row[p] - 1)
        return model_param

    def dose_to_model(self, p_wt=1):
        conc_drug = self.dose_to_conc()
        act = self.conc_moa_effect(conc_drug, self.receptor_binding(conc_drug))
        model_param = self.act_to_model_param(act, p_wt)
        return model_param
        
