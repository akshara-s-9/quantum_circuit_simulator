import numpy as np
import matplotlib.pyplot as plt
#import streamlit as st

#represent a single qubit
qubit=np.array([1/np.sqrt(2)+0j, 1j/np.sqrt(2)], dtype=complex)
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

def hadamard(qubit):
    gate=[[1/np.sqrt(2), 1/np.sqrt(2)],
          [1/np.sqrt(2), -1/np.sqrt(2)]]
    return gate@qubit

def plot_probabilities(qubit):
    probs = np.abs(qubit) ** 2
    labels = [f"|{i}>" for i in range(len(qubit))]
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Measurement probabilities")
    plt.show()

plot_probabilities(qubit)