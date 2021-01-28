#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:42:02 2021

@author: caiafa
"""


import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt


def mpPDF(variancia, razao_T_N, qts_pontos):
    '''
    Gera a Função de densidade de probabilidade archenko-Pastur

    Parameters
    ----------
    variancia : float
        Variância da função.
    razao_T_N : float
        Onde T é o número de linhas e N é o número de colunas.
    qts_pontos : TYPE
        Quantidade de pontos da PDF

    Returns
    -------
    pd.Series -> Função de densidade de probabilidade Archenko-Pastur

    '''
    if isinstance(variancia, np.ndarray):
        if variancia.shape == (1,):
            variancia = variancia[0]
            
    auto_valor_minimo = variancia * (1 - (1/razao_T_N)**0.5)**2
    auto_valor_maximo = variancia * (1 + (1/razao_T_N)**0.5)**2
    autos_valores = np.linspace(auto_valor_minimo, auto_valor_maximo,
                                qts_pontos)
    
    pdf = razao_T_N/(2*np.pi*variancia*autos_valores)*((auto_valor_maximo-autos_valores)*(autos_valores-auto_valor_minimo))**0.5
    
    return pd.Series(pdf, index=autos_valores)

def fitKDE(observation, bWidth= 0.25, kernel='gaussian', x= None):
    '''
    Ajusta o kernel a uma serie de observações e deriva a probabilidade
    das observações.

    Parameters
    ----------
    observation : np.array
        observações para ajustar o kernel. Normalmente é a diagonal de
        auto valores.
    bWidth : float, optional
        DESCRIPTION. A langura de banda do kernel, por padrão é 0.25.
    kernel : str, optional
        DESCRIPTION. O kernel usado para fazer o ajuste.
        Podem ser usados [‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’]
        O padrão é 'gaussian'.
    x : np.array, optional
        É o conjunto de dados que vai receber o ajuste.

    Returns
    -------
        pd.Series -> Empirical PDF

    '''
    if(len(observation.shape) ==1):
        observation = observation.reshape(-1,1)
        
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(observation)
    
    if (x == None):
        x = np.unique(observation).reshape(-1,1)
    if (len(x.shape)==1):
        x = x.reshape(-1,1)
    logProb = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec = eVal[indices],eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal,eVec




# Generating a random matrix
x = np.random.normal(size=(10000,1000))
# Getting eigenvalues and eigenvectors
eVal0, eVec0 = getPCA(np.corrcoef(x, rowvar=0))
# Marchenko-Pastur pdf
pdf0 = mpPDF(1., razao_T_N=x.shape[0]/float(x.shape[1]), qts_pontos=1000)
# Empirical pdf
pdf1= fitKDE(np.diag(eVal0), bWidth=0.01)

# Plotting results
ax = pdf0.plot(title="Marchenko-Pastur Theorem", label="Marchenko-Pastur", color="blue")
pdf1.plot(label="Empirical Value", color="red")
ax.set(xlabel="λ", ylabel="prob[λ]")
ax.legend(loc="upper right")





    

        
    
    
    