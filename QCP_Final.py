# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:13:54 2023

@author: ajaj3
"""

#External Packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import copy 
from numpy.random import randint
from fractions import Fraction
from typing import List,Tuple
# qbit Class
# it creates a qbit in the |0> state. 
# Properties
# state: gives an array with the amplitude of the state |1> and |0> respectively 

class qbit:
    def __init__(self, state:List[complex]=[1,0]):
        self.state = np.array(state,dtype=complex)
        
# Quantum Register Class
# it creates a quantum register of n (length) qbits  all in the state |0>. 
# Properties
# length= number of qbit in the quantum register (type:int)
# tensor= state vector of que the quantum register (type:array)

class quantum_reg:    
    def __init__(self,length:int):
        self.length=length
        qb_array=[]
        for i in range (length):
            qb_array.append(qbit())
        self.qb_array=np.array(qb_array)
        self.tensor_prod_priv()
    
    #tensor_prod_priv : method that creates the initial state vector of the system by doing tensor product
    #of the individual qbits.     
    def tensor_prod_priv(self):
        self.tensor=self.qb_array[0].state;
        for qb in self.qb_array[1:]:
            temp=[]
            for i in range(np.size(qb.state)):
                for j in range(np.size(self.tensor)):
                    temp.append(self.tensor[j]*qb.state[i])
            self.tensor=np.array(temp) / np.linalg.norm(np.array(temp))
            
            
#--------------------------- 1 QBit Gates ---------------------------------------#          
    
    #h:  Hadamard Gate method which creates the matrix of the gate whichc acts on the state n (pos) of the quantum register
    # This method applies the Hadamard gate to the state vector of the quantum register.
    #Parameters:
    #pos:  position of the qbit to which the h gate is applied (type:int)
    #return: an array with the matrix corresponding to the Hadamard gate.
    
    def h(self,pos:int):
        matrix=np.zeros([2**self.length,2**self.length], dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;
            pos1=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)
            matrix[pos1,i]=(-1)**bin_i[-1-pos]
            pos2=bin_i;
            pos2[-pos-1]=1-bin_i[-1-pos]
            pos2=int(''.join(np.array(pos2,dtype=str).tolist()), 2)               
            matrix[pos2,i]=1

        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix

       
    def p(self,pos:int,theta:float,conj:int=0):
        matrix=np.zeros([2**self.length,2**self.length],dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;
            if bin_i[-1-pos]==0:
                pos1=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)
                matrix[pos1,i]=1
            else: 
                pos1=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)
                matrix[pos1,i]=np.exp(1j*theta*((-1)**conj))

        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix
    
    def t(self,pos:int,conj:int=0):
        return self.p(pos,np.pi*0.25,conj)
    
    def s(self,pos:int,conj:int=0):
        return self.p(pos,np.pi*0.5,conj)
    
    def Rz(self,pos:int,theta:float,conj:int=0):
        matrix=np.zeros([2**self.length,2**self.length], dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base; 
            pos1=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)
            matrix[pos1,i]=np.exp((-1)**(1-bin_i[-1-pos])*1j*((-1)**(conj))*theta/(2))

        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix
    
    def Ry(self,pos:int,theta:float=0):
        matrix=np.zeros([2**self.length,2**self.length], dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;
            pos1=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)
            matrix[pos1,i]=np.cos(theta/2)
            pos2=bin_i;
            pos2[-pos-1]=1-bin_i[-1-pos]
            pos2=int(''.join(np.array(pos2,dtype=str).tolist()), 2)               
            matrix[pos2,i]=(-1)**bin_i[-1-pos]*np.sin(theta/2)
            
        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix
    

    #///////////////////// Pauli Gates //////////////////////#
        
    #z: Pauli z Gate method which creates the matrix of the gate which acts on the state n (pos) of the quantum register
    # This method applies the z gate to the state vector of the quantum register.
    #Parameters:
    #pos:  position of the qbit to which the z gate is applied (type:int)
    #return: an array with the matrix corresponding to the Z gate.
    
    def z(self,pos:int):
        return self.p(pos,np.pi)
    
    
    def x(self, pos:int):
        matrix=self.h(pos)
        matrix=matrix.dot(self.z(pos))
        matrix=matrix.dot(self.h(pos))

        return matrix
    
    def y(self, pos:int):
        matrix=self.s(pos)
        matrix=matrix.dot(self.x(pos))
        matrix=matrix.dot(self.s(pos,1))
        return matrix
    
#--------------------------- 2 QBit Gates ---------------------------------------#   

    def cx(self,control:int,target:int):
        matrix=np.zeros([2**self.length,2**self.length],dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;
            bin_i[-1-target]=1*bin_i[-1-control]+((-1)**bin_i[-1-control])*bin_i[-1-target]
            pos3=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)               
            matrix[pos3,i]=1

        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix    
    
    def cp(self,control:int,target:int,theta:int,conj:int=0):
        matrix=np.zeros([2**self.length,2**self.length],dtype=complex)
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;
            pos3=int(''.join(np.array(bin_i,dtype=str).tolist()), 2)  
            matrix[pos3,i]=np.exp(((-1)**conj)*bin_i[-1-target]*bin_i[-1-control]*1j*theta)

        resp=matrix.dot(self.tensor)
        self.tensor=resp/ np.linalg.norm(np.array(resp))
        matrix=matrix/ np.linalg.norm(np.array(resp))
        return matrix 
    
    def ck(self,control:int,target:int,k:int=1,conj:int=0):
        matrix=self.cp(control,target,2*np.pi/(2**k))
        return matrix 
        
    def cz(self, control:int, target:int):
        matrix=self.h(target)
        matrix=matrix.dot(self.cx(control,target))
        matrix=matrix.dot(self.h(target))
        return matrix
    
    def swap(self, qb1:int, qb2:int):
        matrix=self.cx(qb1,qb2);
        matrix=matrix.dot(self.cx(qb2,qb1))
        matrix=matrix.dot(self.cx(qb1,qb2))
        return matrix
    
    
#---------------------- Other Gates ------------------------#

    #///////////////////// Quantum Fourier Transform //////////////////////#

    def cft(self,n: int):
        matrix=np.diag(np.ones(2**self.length,dtype=complex))
        for i in range(n):
            matrix=matrix.dot(self.h(n-1-i))
            for j in range(n-1-i):
                matrix=matrix.dot(self.ck(j,n-1-i,n-i-j))
        for qb in range(n//2):
             matrix=matrix.dot(self.swap(qb, n-qb-1))
        return matrix
    
    def icft(self,n: int):
        matrix=np.diag(np.ones(2**self.length,dtype=complex))
        for qb in range(n//2):
             matrix=matrix.dot(self.swap(qb, n-qb-1))
        for i in range(n):
            for j in range(i):
                matrix=matrix.dot(self.ck(j,i,i-j+1,1))
            matrix=matrix.dot(self.h(i))
        return matrix
    
    def toffoli(self,c1:int,c2:int,t:int):
        matrix=np.diag(np.ones(2**self.length))
        matrix=matrix.dot(self.h(t))
        matrix=matrix.dot(self.cx(c2,t))
        matrix=matrix.dot(self.t(t,1))
        matrix=matrix.dot(self.cx(c1,t))
        matrix=matrix.dot(self.t(t))
        matrix=matrix.dot(self.cx(c2,t))
        matrix=matrix.dot(self.t(t,1))
        matrix=matrix.dot(self.cx(c1,t))
        matrix=matrix.dot(self.t(t))
        matrix=matrix.dot(self.t(c2))
        matrix=matrix.dot(self.cx(c1,c2))
        matrix=matrix.dot(self.h(t))
        matrix=matrix.dot(self.t(c2,1))
        matrix=matrix.dot(self.t(c1))
        matrix=matrix.dot(self.cx(c1,c2))

        return matrix

    def ccz(self, reg, control1:int,control2:int, target:int):
        matrix=reg.cx(control1,target)
        matrix=matrix.dot(reg.t(target,1))
        matrix=matrix.dot(reg.cx(control2,target))
        matrix=matrix.dot(reg.t(target))
        matrix=matrix.dot(reg.cx(control1,target))
        matrix=matrix.dot(reg.t(target,1))
        matrix=matrix.dot(reg.cx(control2,target))
        matrix=matrix.dot(reg.t(target))
        matrix=matrix.dot(reg.t(control1))
        matrix=matrix.dot(reg.cx(control2,control1))
        matrix=matrix.dot(reg.t(control1,1))
        matrix=matrix.dot(reg.cx(control2,control1))
        matrix=matrix.dot(reg.t(control2))

        return matrix

    def cccz(self, reg, control1:int,control2:int,control3:int, target:int):
        theta=np.pi/8
        #--------------------------#
        matrix=reg.p(control1,theta)

        #--------------------------------#
        matrix=matrix.dot(reg.cx(control1,control2))
        matrix=matrix.dot(reg.p(control2,theta,1))
        matrix=matrix.dot(reg.cx(control1,control2))
        matrix=matrix.dot(reg.p(control2,theta))
        #--------------------------------#
        matrix=matrix.dot(reg.cx(control2,control3))
        matrix=matrix.dot(reg.p(control3,theta,1))
        matrix=matrix.dot(reg.cx(control1,control3))
        matrix=matrix.dot(reg.p(control2,theta,0))
        matrix=matrix.dot(reg.cx(control2,control3))
        matrix=matrix.dot(reg.p(control3,theta,1))
        matrix=matrix.dot(reg.cx(control1,control3))
        matrix=matrix.dot(reg.p(control2,theta,0))

        #---------------------------------#    
        matrix=matrix.dot(reg.cx(control3,target))
        matrix=matrix.dot(reg.p(target,theta,1))
        matrix=matrix.dot(reg.cx(control1,target))    
        matrix=matrix.dot(reg.p(target,theta))
        matrix=matrix.dot(reg.cx(control2,target))  
        matrix=matrix.dot(reg.p(target,theta,1)) 
        matrix=matrix.dot(reg.cx(control1,target))
        matrix=matrix.dot(reg.p(target,theta))

        matrix=matrix.dot(reg.cx(control3,target))
        matrix=matrix.dot(reg.p(target,theta,1))
        matrix=matrix.dot(reg.cx(control1,target))    
        matrix=matrix.dot(reg.p(target,theta))
        matrix=matrix.dot(reg.cx(control2,target))  
        matrix=matrix.dot(reg.p(target,theta,1)) 
        matrix=matrix.dot(reg.cx(control1,target))
        matrix=matrix.dot(reg.p(target,theta))

        return matrix

    def FADDa(self, a:int):
        n=self.length
        conj=0
        if a<0:conj=1;a=np.abs(a)
        binary=np.flip(np.asarray([*(format(a, 'b').zfill(n))],dtype=int));
        matrix=np.diag(np.ones(2**n))
        for i in range(n):
            for j in range(n-i):
                if binary[j]==1:
                    matrix=matrix.dot(self.p(i,2*np.pi/(2**(n-i-j)),conj));            
        return matrix
    
    #///////////////////// Qbit Measument  //////////////////////#

    def projection(self,qb:int,state:int):
        matrix=np.diag(np.zeros(2**self.length))
        for i in range(2**self.length):
            base=np.zeros(self.length,dtype=int)
            bin_i=[*(str(bin(i))[2:])]
            base[-len(bin_i):]=bin_i
            bin_i=base;               
            matrix[i,i]= int(bin_i[-1-qb]==state);
            
        return matrix
    
    def measure(self, qb:int):
        projected=self.projection(qb,0).dot(self.tensor)
        norm_projected= np.linalg.norm(projected) 
        if np.random.random()<norm_projected**2: 
            self.tensor=projected/norm_projected
            return 0
        else: 
            projected=self.projection(qb,1).dot(self.tensor)
            self.tensor=projected/np.linalg.norm(projected)
            return 1
        
class Algorithms:    
    
    def __init__(self, register):
        self.reg = register
        
    #grover_oracle_2qbit: method that applies an oracle to a given quantum register for a certain specified state.
    # This method applies a grover's oracle to a 2 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's oracle matrix is going to be applied (type:quantum_reg)
    # state: List of string with the states which want to be found with grover's algorithm. 
    # return: an array with the matrix corresponding to the grover's oracle gate matrix representation.

    def grover_oracle_2qbit(self,states:list):
        matrix=np.diag([1,1,1,1])
        for state in states:
            binary=np.array([*(state)],dtype=int)
            positions=np.where(binary==0)[0];
            for i in range(0,np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))
            matrix=matrix.dot(self.reg.cz(0,1))
            for i in range(np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))
        return matrix

    #grover_amplification_2qbit: method that applies a  grover's amplification matrix to a given 2 qbit quantum register for a certain specified state.
    # This method applies a grover's amplification gate to a 2 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's amplification matrix is going to be applied (type:quantum_reg)
    # return: an array with the matrix corresponding to the grover's amplification gate matrix representation.

    def grover_amplification_2qbit(self):
        matrix=np.diag([1,1,1,1])
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));

        matrix=matrix.dot(self.reg.cz(1,0))

        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));

        return matrix 
    
    
    #----------------------------------------3 Qubit Grovers -----------------------------------------------------------
    
    #grover_oracle_3qbit: method that applies an oracle to a given quantum register for a certain specified state.
    # This method applies a grover's oracle to a 3 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's oracle matrix is going to be applied (type:quantum_reg)
    # state: List of string with the states which want to be found with grover's algorithm. 
    # return: an array with the matrix corresponding to the grover's oracle gate matrix representation.
    def grover_oracle_3qbit(self,values:list):
        matrix=np.diag(np.ones(2**3))
        for state in values:
            binary=np.array([*(state)],dtype=int)
            positions=np.where(binary==0)[0];
            for i in range(0,np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))
            matrix=matrix.dot(self.reg.ccz(self.reg,2,1,0))
            for i in range(np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))
        return matrix

    #grover_amplification_3qbit: method that applies a  grover's amplification matrix to a given 3 qbit quantum register for a certain specified state.
    # This method applies a grover's amplification gate to a 3 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's amplification matrix is going to be applied (type:quantum_reg)
    # return: an array with the matrix corresponding to the grover's amplification gate matrix representation.
    def grover_amplification_3qbit(self):
        matrix=np.diag(np.ones(2**3))
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));

        matrix=matrix.dot(self.reg.ccz(self.reg,2,1,0))

        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));

        return matrix 
    
    
    
    #------------------------------------------4 Qubit Grovers ---------------------------------------------------
    
    #grover_oracle_4qbit: method that applies an oracle to a given quantum register for a certain specified state.
    # This method applies a grover's oracle to a 4 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's oracle matrix is going to be applied (type:quantum_reg)
    # state: List of string with the states which want to be found with grover's algorithm. 
    # return: an array with the matrix corresponding to the grover's oracle gate matrix representation.
    def grover_oracle_4qbit(self, values:list):
        matrix=np.diag(np.ones(2**4))
        for state in values:
            binary=np.array([*(state)],dtype=int)
            positions=np.where(binary==0)[0];
            for i in range(0,np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))
            #print(self)
            matrix=matrix.dot(self.reg.cccz(self.reg, 3,2,1,0))
            #matrix=matrix.dot(cnz(reg,[3,2,1],0))

            for i in range(np.size(positions)):
                pos=(np.size(binary)-1)-positions[i]
                matrix=matrix.dot(self.reg.x(pos))

        return matrix

    #grover_amplification_4qbit: method that applies a  grover's amplification matrix to a given 4 qbit quantum register for a certain specified state.
    # This method applies a grover's amplification gate to a 4 qbit quantum register.
    # Parameters:
    # reg: quantum register to which the grover's amplification matrix is going to be applied (type:quantum_reg)
    # return: an array with the matrix corresponding to the grover's amplification gate matrix representation.
    def grover_amplification_4qbit(self):
        matrix=np.diag(np.ones(2**4))
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));

        matrix=matrix.dot(self.reg.cccz(self.reg,3,2,1,0))
        #matrix=matrix.dot(cnz(reg,[3,2,1],0))
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.x(qb));
        for qb in range(self.reg.length):
            matrix=matrix.dot(self.reg.h(qb));

        return matrix 
    
    #------------------------------------Shor's Algorithm----------------------------------------------------
    
    

    def gcd(self, a, b):
        while b != 0:
            t = copy.copy(b)
            b = a% b
            a = copy.copy(t)
        return a 

    def U(self, r, a:int,N:int,counting_qb:int):
        repetitions=1;
        matrix=np.diag(np.ones(2**r.length))
        #print(counting_qb)
        for i in range(counting_qb):
            for j in range(repetitions):
                matrix=matrix.dot(r.cp(counting_qb,i,2*np.pi*a/((2**j)*N)));
            repetitions*=2  
        return matrix


    #Shors: method that applies shors algorithm to factorize numbers.
    # Parameters:
    # N: Number which is going to be factorized.
    # return: an array with two factors of N. 

    def Shors(self, N, n, m):
        #N = int(input("What number do you wish to factorise? "))
      
    #Choosing a number between 2 and N-1  which is not already a factor of N
        a=np.random.randint(2, N-1)
        while self.gcd(a,N)!=1:
            a=np.random.randint(2, N-1)
        
    #Determining the number of input and output qbits.
        #n = int(np.ceil(np.log2(N)))
        #m = 1
        
        
    #Determining the number of experiments and measurements which are going to be done    
        measures=[]
        tot_measures=50
        for k in range(tot_measures):  
            reg = quantum_reg(n+m)
            for i in range(n):
                reg.h(i);
            #Initializing the input qbits in state |1>
            reg.x(n)

            #Applying the controlled U^j gate to the input qbit  for each output qbit present.
            self.U(reg, a,N,n);

            #Applying the inverse Quantum Fourier Transform. 
            reg.icft(n);

            #Mearusing the state of the output qbits and storing the result
            value=[]
            for i in range(n):
                value.append(reg.measure(i));

            measures.append(int(''.join(np.flip(np.array(value,dtype=str).tolist())), 2))


        #Determining the possible values of r and the possible factors
        labels, counts = np.unique(measures, return_counts=True)      
        #plt.bar(labels, counts, align='center')
        #plt.title("Likely state location of Period")
        #plt.gca().set_xticks(labels)
        #plt.xticks(np.arange(0,2**(n), 2))
        #plt.ylabel('Counts')
        #plt.xlabel('State')

        fractions = []
        for i in range(len(labels)):
            if counts[i]> tot_measures*0.1:
                f = Fraction(labels[i]/2**n).limit_denominator(N)
                r = f.denominator
                fraction = self.gcd(N, int(a**(r/2)) +1)
                fractions.append(fraction)

        #Validating the possible factors found and returning the result. 
        r = 0
        evaluations=0
        while r == 0 and evaluations<N:
            a = randint(2, N)
            r = self.gcd(a, N)
            for i in range(len(fractions)):
                evaluations+=1
                if r != 1 and r != N and r == fractions[i]:
                    break
                if i == len(fractions)-1:
                    r = 0

        if r < 2 or r == N :
            return "Failed to find factors"
        else:
            return r, N // r

import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QVBoxLayout, QApplication, QWidget, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit

class GUI(QWidget):
    """
    This class is the main simulator. From here all functionality can be selected.
    """
    def __init__(self):
        super().__init__()
        #Calling the main page interface
        self.QCUI()

    def QCUI(self):
        #Creating the window
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Simulator')

        #Buttons for each algorithm, plus an extra that takes the user to a help page
        Grovers = QPushButton('Grovers', self)
        Grovers.clicked.connect(self.Grov_alg)
        Grovers.move(50, 50)

        Shor_search = QPushButton('Shor Search', self)
        Shor_search.clicked.connect(self.Shor_search_alg)
        Shor_search.move(150, 50)


        Help = QPushButton('Help', self)
        Help.clicked.connect(self.Help_alg)
        Help.move(150, 100)

        #Show the interface
        self.show()

    #All the methods we need for the emulator
    def Grov_alg(self):
        self.grov_window = GrovWindow()

    def Shor_search_alg(self):
        self.shor_window = ShorWindow()       

    def Help_alg(self):
        self.help_window = HelpWindow()
        
class ShorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.Win()
        
    def Win(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle("Shor's Algorithm")
        
        ShorQ = QPushButton("Run Shor's", self)
        ShorQ.clicked.connect(self.ShorQ_click)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(ShorQ)
        
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.show()
        
    def ShorQ_click(self):
        #N = int(input("Which number do you wish to factorise"))
        
        N, ok = QInputDialog.getInt(self, "Enter an Integer ", "Number: ", min=0)
        if not ok:
            N = None
        
        n = int(np.ceil(np.log2(N)))
        m = 1

        Algs = Algorithms(quantum_reg(n+m))
        f = False
        while f == False:
            if Algs.Shors(N, n, m) == (5, 3):
                answer = QMessageBox()
                answer.setText("(5, 3)")
                answer.exec_()
                f = True
            elif Algs.Shors(N, n, m) == (3, 5):
                answer = QMessageBox()
                answer.setText("(3, 5)")
                answer.exec_()
                f = True

class GrovWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.Win()

    def Win(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle("Grover's Algorithm")
        
        TwoQ = QPushButton("2-Qubit Search", self)
        ThreeQ = QPushButton("3-Qubit Search", self)
        FourQ = QPushButton("4-Qubit Search", self)
        
        TwoQ.clicked.connect(self.TwoQ_click)
        ThreeQ.clicked.connect(self.ThreeQ_click)
        FourQ.clicked.connect(self.FourQ_click)
        
        button_layout = QVBoxLayout()
        button_layout.addWidget(TwoQ)
        button_layout.addWidget(ThreeQ)
        button_layout.addWidget(FourQ)
        
        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.show()
        
    def TwoQ_click(self):
        reg=quantum_reg(2)
        Algs = Algorithms(reg)
        Algs.reg.tensor
        
        for qb in range(reg.length):
            Algs.reg.h(qb);
        Algs.reg.tensor
         
        N, ok = QInputDialog.getText(self, "Enter a State ", "Number: ")
        
        if not ok:
            N = None
        
        Algs.grover_oracle_2qbit([N])
        Algs.reg.tensor
        Algs.grover_amplification_2qbit()
        
        
        answer = QMessageBox()
        answer.setText(str(Algs.reg.tensor))
        answer.exec_()
        
    def ThreeQ_click(self):
        reg=quantum_reg(3)
        Algs = Algorithms(reg)
        Algs.reg.tensor
        
        for qb in range(Algs.reg.length):
            Algs.reg.h(qb);
        Algs.reg.tensor

        N, ok = QInputDialog.getText(self, "Enter a State ", "Number: ")
        
        if not ok:
            N = None
        
        N= [N]
        Algs.grover_oracle_3qbit(N)
        Algs.reg.tensor
    
        Algs.grover_amplification_3qbit()
        Algs.reg.tensor
    
        Algs.grover_oracle_3qbit(N)
        Algs.reg.tensor
        Algs.grover_amplification_3qbit()
        answer = QMessageBox()
        answer.setText(str(Algs.reg.tensor))
        answer.exec_()

    def FourQ_click(self):
        reg=quantum_reg(4)
        Algs = Algorithms(reg)
        Algs.reg.tensor
        
        for qb in range(Algs.reg.length):
            Algs.reg.h(qb);
        Algs.reg.tensor

        N, ok = QInputDialog.getText(self, "Enter a State ", "Number: ")
        
        if not ok:
            N = None
        
        N = [N]
        Algs.grover_oracle_4qbit(N)
        Algs.reg.tensor

        Algs.grover_amplification_4qbit()
        Algs.reg.tensor

        Algs.grover_oracle_4qbit(N)
        Algs.reg.tensor

        Algs.grover_amplification_4qbit()
        Algs.reg.tensor

        Algs.grover_oracle_4qbit(N)
        Algs.reg.tensor

        Algs.grover_amplification_4qbit()
        
        answer = QMessageBox()
        answer.setText(str(Algs.reg.tensor))
        answer.exec_()
        
#Open up the help page
class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.Win()

    def Win(self):
        
        #Creating the help page
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Help')

        #Create the help buttons
        Q = QPushButton("Qubit", self)
        R = QPushButton("Register", self)
        H = QPushButton("H-gate", self)
        X = QPushButton("X-gate", self)
        Y = QPushButton("Y-gate", self)
        Z = QPushButton("Z-gate", self)
        CNOT = QPushButton("CNOT-gate", self)
        T = QPushButton("T-gate", self)
        Grov = QPushButton("Grovers Alogorithm", self)
        Shor = QPushButton("Shors Algorithm", self)
        Err = QPushButton("Error Correction", self)

        #Connecting buttons to their respective information text box
        Q.clicked.connect(self.Q_click)
        R.clicked.connect(self.R_click)
        H.clicked.connect(self.H_click)
        X.clicked.connect(self.X_click)
        Y.clicked.connect(self.Y_click)
        Z.clicked.connect(self.Z_click)
        CNOT.clicked.connect(self.CNOT_click)
        T.clicked.connect(self.T_click)
        Grov.clicked.connect(self.Grov_click)
        Shor.clicked.connect(self.Shor_click)
        Err.clicked.connect(self.Err_click)

        #creating a layout for the buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(Q)
        button_layout.addWidget(R)
        button_layout.addWidget(H)
        button_layout.addWidget(X)
        button_layout.addWidget(Y)
        button_layout.addWidget(Z)
        button_layout.addWidget(CNOT)
        button_layout.addWidget(T)
        button_layout.addWidget(Grov)
        button_layout.addWidget(Shor)
        button_layout.addWidget(Err)
        
        #placing the layout on the empty window
        layout = QVBoxLayout()
        layout.addLayout(button_layout)

        #showing the window
        self.setLayout(layout)
        self.show()

    #When a button is pressed, it will display a brief description about the function/theory
    def Q_click(self):
        QMessageBox.information(self, "Qubit", "A qubit is the unit of quantum information. It exists in two states at once, which means it can perform multiple calculations at the same time.")

    def R_click(self):
        QMessageBox.information(self, "Register", "A quantum register is a collection of qubits. The number of qubits determines the amount of information that can be stored. Storing qubits in a register allows for them to be measured and manipulated independently")

    def H_click(self):
        QMessageBox.information(self, "Hadamard Gate", "The hadamard gate is applied to individual qubits. It forces it into superposition where the two basis states have equal probability. See more: https://qiskit.org/textbook/ch-states/single-qubit-gates.html#hgate")
                                
    def X_click(self):
        QMessageBox.information(self, "X Gate", "The X gate flips the state of a qubit, i.e from 0 to 1. Also known as a NOT gate. See more: https://qiskit.org/textbook/ch-states/single-qubit-gates.html#xgate")
                                
    def Y_click(self):
        QMessageBox.information(self, "Y Gate", "The Y gate rotates the state of a qubit around the y-axis of the Bloch Sphere. This gives a combination of basis states. See more: https://qiskit.org/textbook/ch-states/single-qubit-gates.html#ynzgatez")
    
    def Z_click(self):
        QMessageBox.information(self, "Z Gate", "The Z gate flips the phase of a qubit, rotating its state around the z-axis of the Bloch Sphere. See more:  https://qiskit.org/textbook/ch-states/single-qubit-gates.html#ynzgatez")
                                
    def CNOT_click(self):
        QMessageBox.information(self, "CNOT Gate", "The CNOT gate is similar to the X gate, except it only applies on the qubit if it is in state |1>. See more: https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html#cnot")
                                
    def T_click(self):
        QMessageBox.information(self, "T-Gate", "The T gate is similar to a Z-gate, however here the qubit state is only rotated a quarter turn around the Z-axis. See more: https://qiskit.org/textbook/ch-states/single-qubit-gates.html#tgate")
                    
    def Grov_click(self):
        QMessageBox.information(self, "Grovers Algorithm", "Grover's algorithm is used to quickly search an unsorted database. It does so considerably faster than classical computer algorithms. It works by creating a superposition of all possible states via a Hadamard gate. It then applies a series of operations to amplify the probability of the desired state. This is done until the desired state is found with high probability. See here: https://qiskit.org/textbook/ch-algorithms/grover.html")

    def Shor_click(self):
        QMessageBox.information(self, "Shors Algorithm", "Shors algorithm is used to find the prime factors of an integer, it works via exploiting periodicity and using a quantum fourier transform. It is very useful for breaking encryption. See more: https://qiskit.org/textbook/ch-algorithms/shor.html")
    
    def Err_click(self):
        QMessageBox.information(self, "Error Correction", "Algorithm used to protect against common types of errors in quantum systems. See more: https://qiskit.org/textbook/ch-quantum-hardware/error-correction-repetition-code.html")
       
if __name__ == '__main__':
 
    app = QApplication(sys.argv)
        
    window = GUI()
        
    sys.exit(app.exec_())
