import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
#import streamlit as st

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
P0 = np.array([[1, 0], [0, 0]])   # |0><0|
P1 = np.array([[0, 0], [0, 1]])   # |1><1|


#input the qubits
def read_qubit(i):
    # keep asking until the user enters a valid, normalised 2-element vector
    while True:
        qubit_input = input(f"Enter qubit {i} as a vector, e.g. [1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j]: ")
        try:
            # evaluate the text as a Python expression, only allowing numpy (np) to be used
            qubit = np.array(eval(qubit_input, {"__builtins__": {}, "np": np}), dtype=complex)
        except Exception:
            print("Could not read that vector, please try again.")
            continue
        if qubit.shape != (2,):
            print("A qubit must have exactly 2 amplitudes, please try again.")
            continue
        # use isclose because floating point maths isn't exact (e.g. 0.5000000000000001)
        if not np.isclose(abs(qubit[0])**2 + abs(qubit[1])**2, 1):
            print("The qubit is not normalised (|a|^2 + |b|^2 must equal 1), please try again.")
            continue
        print(f"Qubit {i} is valid: {qubit}")
        return qubit


#qubit gates
def num_qubits_in(state):
    # a state of n qubits has 2^n amplitudes
    return int(np.log2(len(state)))

def apply_single_qubit_gate(gate, state, targets=0):
    # apply a 2x2 gate to one or more qubits of a (possibly multi-qubit) state
    # targets can be a single qubit index or a list of indices; qubit 0 is the leftmost bit
    num_qubits = num_qubits_in(state)
    if isinstance(targets, int):
        targets = [targets]
    for t in targets:
        if not 0 <= t < num_qubits:
            raise ValueError(f"target qubit {t} is out of range for a {num_qubits}-qubit state")
    # build gate ⊗ I ⊗ ... with the gate in each target position and identity elsewhere
    full_gate = reduce(np.kron, [gate if i in targets else I for i in range(num_qubits)])
    return full_gate @ state

def pauli_x(state, targets=0):
    gate = np.array([[0, 1],
                     [1, 0]])
    return apply_single_qubit_gate(gate, state, targets)

def pauli_y(state, targets=0):
    gate = np.array([[0, -1j],
                     [1j, 0]])
    return apply_single_qubit_gate(gate, state, targets)

def pauli_z(state, targets=0):
    gate = np.array([[1, 0],
                     [0, -1]])
    return apply_single_qubit_gate(gate, state, targets)

def hadamard(state, targets=0):
    gate = np.array([[1, 1],
                     [1, -1]]) / np.sqrt(2)
    return apply_single_qubit_gate(gate, state, targets)

#probabilities and measurement
def get_probabilities(qubit):
    probs = np.abs(qubit) ** 2
    return probs

def measure(state):
    # pick one basis state at random, weighted by its probability, and return it as a bitstring e.g. "101"
    probs = get_probabilities(state)
    index = np.random.choice(len(state), p=probs)
    return f"{index:0{num_qubits_in(state)}b}"

#multiple qubits
def tensor_product(qubits):
    return  reduce(np.kron, qubits)

#two-qubit gates
# basis order from tensor_product(qubit1, qubit2): |00>, |01>, |10>, |11>
# qubit 0 = qubit1 (left bit), qubit 1 = qubit2 (right bit)

def cnot(state, control=0, target=1):
    # if control is |0> do nothing, if control is |1> apply X to target
    if control == 0 and target == 1:
        gate = np.kron(P0, I) + np.kron(P1, X)
    elif control == 1 and target == 0:
        gate = np.kron(I, P0) + np.kron(X, P1)
    else:
        raise ValueError("control and target must be 0 and 1, and different")
    return gate @ state

#visualization
def plot_probabilities(state):
    probs = get_probabilities(state)
    n = num_qubits_in(state)
    labels = [f"|{i:0{n}b}>" for i in range(len(state))]
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

#main program
while True:
    try:
        n = int(input("Enter the number of qubits: "))
        if n >= 1:
            break
    except ValueError:
        pass
    print("Please enter a whole number of 1 or more.")

qubits = [read_qubit(i) for i in range(n)]

# combine all the qubits into one state: qubit 0 is the leftmost bit
state = tensor_product(qubits)
print(f"Combined {n}-qubit state:")
for index, amplitude in enumerate(state):
    print(f"  |{index:0{n}b}>: {amplitude}")

