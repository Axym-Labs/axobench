TITLE Deterministic GABA_A receptor for AxoBench teachers

COMMENT
Deterministic subset of ProbUDFsyn2 used by the Hay source-teacher generator
when Use=1, Dep=0, Fac=0, u0=0. Each event always increments a
dual-exponential inhibitory conductance state by the NetCon weight.

The dual-exponential scheme follows NEURON's exp2syn.mod. Parameter values
follow ProbUDFsyn2.mod (Blue Brain Project, as distributed with Hay et al.
2011 and NeuronIO); none of its stochastic release code is used.

SPDX-License-Identifier: BSD-3-Clause
Copyright (c) 2018, Michael Hines (NEURON exp2syn.mod)
Copyright (c) 2026, Axym Labs (modifications)
Licence text: LICENSES/NEURON-BSD-3-Clause.txt in the axobench repository.
ENDCOMMENT

NEURON {
    POINT_PROCESS DetGABA
    RANGE tau_r, tau_d
    RANGE i, g, e, gmax
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    tau_r = 0.2 (ms)
    tau_d = 8.0 (ms)
    e = -80 (mV)
    gmax = 0.001 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    factor
}

STATE {
    A
    B
}

INITIAL {
    LOCAL tp

    A = 0
    B = 0
    tp = (tau_r*tau_d)/(tau_d-tau_r)*log(tau_d/tau_r)
    factor = 1/(-exp(-tp/tau_r)+exp(-tp/tau_d))
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = gmax*(B-A)
    i = g*(v-e)
}

DERIVATIVE state {
    A' = -A/tau_r
    B' = -B/tau_d
}

NET_RECEIVE (weight) {
    A = A + weight*factor
    B = B + weight*factor
}
