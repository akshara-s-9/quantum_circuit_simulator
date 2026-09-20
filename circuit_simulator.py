import numpy as np
import matplotlib.pyplot as plt
#import streamlit as st

#represent a single qubit
qubit=np.array([1+0j, 0+1j], dtype=complex)
print(qubit)

if abs(qubit[0])**2 + abs(qubit[1])**2 == 1:
    print("The qubit is valid.")

#qubit gates
def pauli_x(qubit):
    gate=[[0, 1],
          [1, 0]]
    return gate@qubit

def pauli_y(qubit):
    gate=[[0, -1j],
          [1j, 0]]
    return gate@qubit

def pauli_z(qubit):
    gate=[[1, 0],
          [0, -1]]
    return gate@qubit

print(pauli_x(qubit))