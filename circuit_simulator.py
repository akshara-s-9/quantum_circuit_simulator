import numpy as np
import matplotlib.pyplot as plt
#import streamlit as st

#represent a single qubit

qubit_input = input("Enter the qubit state, e.g. [1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j]: ")
# evaluate the text as a Python expression, only allowing numpy (np) to be used
qubit = np.array(eval(qubit_input, {"__builtins__": {}, "np": np}), dtype=complex)
print(qubit)

# use isclose because floating point maths isn't exact (e.g. 0.5000000000000001)
if np.isclose(abs(qubit[0])**2 + abs(qubit[1])**2, 1):
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

#probabilities and measurement
def get_probabilities(qubit):
    probs = np.abs(qubit) ** 2
    return probs

def measure(qubit):
    probs = get_probabilities(qubit)
    result = np.random.choice([0, 1], p=probs)
    return result

#visualization
def plot_probabilities(qubit):
    probs = get_probabilities(qubit)    
    labels = [f"|{i}>" for i in range(len(qubit))]
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Measurement probabilities")
    plt.show()

def plot_bloch_sphere(qubit):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    rx = 2 * np.real(qubit[0] * np.conj(qubit[1]))
    ry = 2 * np.imag(qubit[0] * np.conj(qubit[1]))
    rz = abs(qubit[0])**2 - abs(qubit[1])**2

    ax.plot([-1, 1], [0, 0], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [-1, 1], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 0], [-1, 1], color='black', linewidth=1)
    ax.quiver(0, 0, 0, rx, ry, rz, color='r', linewidth=2)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
        axis.line.set_color((1, 1, 1, 0))

    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for value in (-1, 0, 1):
        ax.text(value, 0, 0, f"{value}", ha='center', va='center')
        ax.text(0, value, 0, f"{value}", ha='center', va='center')
        ax.text(0, 0, value, f"{value}", ha='center', va='center')

    ax.text(1.12, 0, 0, 'X', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(0, 1.12, 0, 'Y', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(0, 0, 1.12, 'Z', fontsize=12, fontweight='bold', ha='center', va='center')

    u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(u.size), np.cos(v))

    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    ax.set_box_aspect([1, 1, 1])
    plt.show()

measurement_result = measure(qubit)
print(f"Measurement result: |{measurement_result}>")

