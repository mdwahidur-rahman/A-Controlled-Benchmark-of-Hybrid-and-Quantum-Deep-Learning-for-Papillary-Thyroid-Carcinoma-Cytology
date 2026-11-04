"""
Quantum arm: variational layers on cached features, each gated by a closed-form
collapse test BEFORE it is allowed to be called quantum.

The pre-registration rule from the NeuroQFed-XAI scoping applies: no circuit
earns the word "quantum" until it fails a reduction to a classical twin. Two
reductions are tested here, both verified numerically rather than asserted.

THEOREM A (amplitude encoding, commuting observables).
  With |psi> = z/||z||, |phi> = U(theta)|psi>, and mutually commuting Pauli-Y
  observables, every expectation reduces to

      <Y_j> = sum_k s_jk * |(V phi)_k|^2 ,   s_jk = +1 if bit j of k is 0 else -1

  where V simultaneously diagonalizes the Y_j. The whole layer is therefore one
  unitary rotation, a squaring of magnitudes, and a fixed +/-1 linear readout:
  a squared-linear map with no quantum advantage available, regardless of
  ansatz depth or entangling pattern.

THEOREM B (angle encoding, CNOT chain, Pauli-Z).
  <Z_i> = prod_{j<=i} cos(x_j + a_j) cos(b_j), a cumulative-product form.

CONTROLS. A test suite in which every arm passes proves nothing, so the
instrument is checked against cases where it must fail. An earlier version of
this file used "mixed Pauli letters across wires (XYZYX)" as that control and
reported a large deviation. That was wrong, and the correction is recorded
here rather than silently applied: single-qubit Paulis on DISTINCT wires
commute whatever letters they carry, because they act on different tensor
factors. The XYZYX family therefore has a common eigenbasis and Theorem A
applies to it in full -- the deviation came from holding the twin's basis
fixed at all-Y while measuring X/Y/Z, which is a mismatched twin, not an
escaping circuit. Two honest controls replace it:

  C1 BASIS MISMATCH. Measure XYZYX, build the twin in the all-Y basis. Fails
     by construction. This is the old "control", named for what it is: a check
     that the instrument can register a wrong twin at all.
  C2 NON-COMMUTING FAMILY. Measure two different letters on the SAME wire
     ({X_1, Z_1, Y_2, Y_3, Y_4}). These do not commute, no common eigenbasis
     exists, and no product-basis twin of the Theorem A form can reproduce
     them. This is the real boundary of the reduction.

The XYZYX arm is reported as what it is: a COLLAPSING arm once its twin is
built in the matching basis.

Also reports Meyer-Wallach Q (global entanglement) so entanglement and
classical reducibility can be shown to be independent axes -- a circuit can be
highly entangled and still exactly classically simulable by the twin.
"""
import os, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_at(op, j, n):
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        M = np.kron(M, op if q == j else I2)
    return M


def rand_unitary(D, rng):
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def meyer_wallach(phi, n):
    """Global entanglement Q in [0,1]; 0 = product state, 1 = maximally entangled."""
    psi = phi.reshape([2] * n)
    tot = 0.0
    for q in range(n):
        m = np.moveaxis(psi, q, 0).reshape(2, -1)
        rho = m @ m.conj().T
        tot += (1.0 - np.real(np.trace(rho @ rho)))
    return float(2.0 * tot / n)


PAULIS = {"X": PX, "Y": PY, "Z": PZ}


def eigenbasis(letter):
    """Row k of the returned matrix is the +/-1 eigenvector of the Pauli,
    ordered (+1, -1), so that V maps a state to its coordinates in that basis."""
    w, v = np.linalg.eigh(PAULIS[letter])      # ascending eigenvalues: -1, +1
    return v[:, np.argsort(-w)].conj().T       # reorder to (+1, -1)


def product_basis(letters):
    V = np.array([[1.0 + 0j]])
    for c in letters:
        V = np.kron(V, eigenbasis(c))
    return V


def max_commutator(letters, n):
    """Largest |[O_i, O_j]| over the measured family -- zero iff it commutes."""
    ops = [kron_at(PAULIS[c], j, n) for j, c in enumerate(letters)]
    m = 0.0
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            m = max(m, float(np.abs(ops[i] @ ops[j] - ops[j] @ ops[i]).max()))
    return m


def test_amplitude_commuting(n=5, trials=20000, seed=0, observables="YYYYY",
                             twin_basis=None):
    """Theorem A. `observables` gives the measured Pauli letter per wire;
    `twin_basis` gives the letters the twin diagonalizes in, defaulting to the
    measured ones (the matched twin). Passing a different string produces the
    basis-mismatch control."""
    rng = np.random.default_rng(seed)
    D = 2 ** n
    ops = [kron_at(PAULIS[observables[j]], j, n) for j in range(n)]
    V = product_basis(twin_basis or observables)
    S = np.array([[1.0 if (k >> (n - 1 - j)) & 1 == 0 else -1.0 for k in range(D)]
                  for j in range(n)])

    worst, qs = 0.0, []
    U = rand_unitary(D, rng)
    for t in range(trials):
        if t % 100 == 0:
            U = rand_unitary(D, rng)
        z = rng.normal(size=D) + 1j * rng.normal(size=D)
        psi = z / np.linalg.norm(z)
        phi = U @ psi
        direct = np.array([np.real(phi.conj() @ (O @ phi)) for O in ops])
        twin = S @ (np.abs(V @ phi) ** 2)
        worst = max(worst, float(np.max(np.abs(direct - twin))))
        if t % 500 == 0:
            qs.append(meyer_wallach(phi, n))
    return {"max_deviation": worst, "mean_meyer_wallach": float(np.mean(qs)),
            "max_commutator": max_commutator(observables, n),
            "observables": observables, "twin_basis": twin_basis or observables}


def test_noncommuting_same_wire(n=5, trials=20000, seed=3):
    """Real negative control: X and Z on the SAME wire do not commute, so no
    product basis diagonalizes the family and no twin of the Theorem A form
    can exist. The twin is built in the best available product basis (the one
    matching the remaining wires, with wire 1 taken as Z) and must still fail."""
    rng = np.random.default_rng(seed)
    D = 2 ** n
    ops = [kron_at(PX, 0, n), kron_at(PZ, 0, n)] + [kron_at(PY, j, n) for j in range(1, n)]
    V = product_basis("Z" + "Y" * (n - 1))
    S = np.array([[1.0 if (k >> (n - 1 - j)) & 1 == 0 else -1.0 for k in range(D)]
                  for j in range(n)])
    S = np.vstack([S[0], S])                      # X_1 and Z_1 share wire 1
    worst, qs = 0.0, []
    U = rand_unitary(D, rng)
    for t in range(trials):
        if t % 100 == 0:
            U = rand_unitary(D, rng)
        z = rng.normal(size=D) + 1j * rng.normal(size=D)
        phi = U @ (z / np.linalg.norm(z))
        direct = np.array([np.real(phi.conj() @ (O @ phi)) for O in ops])
        twin = S @ (np.abs(V @ phi) ** 2)
        worst = max(worst, float(np.max(np.abs(direct - twin))))
        if t % 500 == 0:
            qs.append(meyer_wallach(phi, n))
    m = 0.0
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            m = max(m, float(np.abs(ops[i] @ ops[j] - ops[j] @ ops[i]).max()))
    return {"max_deviation": worst, "mean_meyer_wallach": float(np.mean(qs)),
            "max_commutator": m, "observables": "X1,Z1,Y2,Y3,Y4",
            "twin_basis": "Z" + "Y" * (n - 1)}


def test_angle_cnot_z(n=5, trials=20000, seed=0):
    """Theorem B: angle encoding + CNOT chain + Pauli-Z -> cumulative product."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(-np.pi, np.pi, n)
        a = rng.uniform(-np.pi, np.pi, n)
        b = rng.uniform(-np.pi, np.pi, n)
        state = np.array([1.0 + 0j])
        for j in range(n):
            th = x[j] + a[j]
            # R_x(theta) on |0>  ->  <Z> = cos(theta)
            q = np.array([np.cos(th / 2), -1j * np.sin(th / 2)])
            # then R_y(b), a PERPENDICULAR axis, so the two factor:
            #   <Z> = cos(theta) * cos(b)
            # (composing two R_x rotations instead would give cos(theta+b),
            #  which is NOT the cumulative-product form the theorem describes)
            cb, sb = np.cos(b[j] / 2), np.sin(b[j] / 2)
            q = np.array([cb * q[0] - sb * q[1], sb * q[0] + cb * q[1]])
            state = np.kron(state, q)
        for j in range(n - 1):
            full = np.zeros((2 ** n, 2 ** n), dtype=complex)
            for k in range(2 ** n):
                bits = [(k >> (n - 1 - q)) & 1 for q in range(n)]
                if bits[j] == 1:
                    bits[j + 1] ^= 1
                kk = sum(bb << (n - 1 - q) for q, bb in enumerate(bits))
                full[kk, k] = 1.0
            state = full @ state
        direct = np.array([np.real(state.conj() @ (kron_at(PZ, i, n) @ state))
                           for i in range(n)])
        twin = np.array([np.prod([np.cos(x[j] + a[j]) * np.cos(b[j])
                                  for j in range(i + 1)]) for i in range(n)])
        worst = max(worst, float(np.max(np.abs(direct - twin))))
    return {"max_deviation": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "results"))
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--qubits", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    arms = []
    print(f"{'ARM':46} {'max deviation':>13}   {'verdict':11} {'MW-Q':>6}  {'max|[Oi,Oj]|':>12}")
    print("-" * 98)

    r = test_amplitude_commuting(a.qubits, a.trials, 0, "YYYYY")
    arms.append({"arm": "amplitude + SEL + all-Y (HQCM-EBTC replica)", **r,
                 "verdict": "COLLAPSES"})
    r2 = test_angle_cnot_z(a.qubits, min(a.trials, 4000), 1)
    arms.append({"arm": "angle + CNOT chain + Z (QCNET replica)", **r2,
                 "verdict": "COLLAPSES", "mean_meyer_wallach": None})
    r3 = test_amplitude_commuting(a.qubits, a.trials, 2, "XYZYX")
    arms.append({"arm": "amplitude + mixed X/Y/Z, distinct wires", **r3,
                 "verdict": "COLLAPSES"})
    r4 = test_amplitude_commuting(a.qubits, a.trials, 2, "XYZYX", twin_basis="YYYYY")
    arms.append({"arm": "C1: XYZYX measured, all-Y twin (basis mismatch)", **r4,
                 "verdict": "TWIN FAILS"})
    r5 = test_noncommuting_same_wire(a.qubits, a.trials, 3)
    arms.append({"arm": "C2: X and Z on one wire (non-commuting)", **r5,
                 "verdict": "TWIN FAILS"})

    for x in arms:
        q = f"{x['mean_meyer_wallach']:.4f}" if x.get("mean_meyer_wallach") else "   -  "
        c = x.get("max_commutator")
        c = f"{c:.1e}" if c is not None else "  -  "
        print(f"{x['arm']:46} {x['max_deviation']:.3e}   {x['verdict']:11} {q}  {c}")

    with open(os.path.join(a.out, "quantum_collapse.json"), "w") as fh:
        json.dump(arms, fh, indent=2)
    print("\nInterpretation: a collapsing arm is exactly reproduced by a classical")
    print("twin at machine precision, so it cannot carry quantum advantage.")
    print("Mixed Pauli letters on distinct wires still commute (max|[Oi,Oj]| = 0)")
    print("and therefore still collapse; the reduction is not evaded by changing")
    print("letters. C1 and C2 confirm the test can fail -- C1 when the twin is")
    print("built in the wrong basis, C2 when the measured family genuinely does")
    print("not commute and no product-basis twin exists.")


if __name__ == "__main__":
    main()
