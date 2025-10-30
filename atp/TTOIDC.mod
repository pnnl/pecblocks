MODEL TTOIDC -- 6 character name limit
INPUT i1
OUTPUT o1,o2,o3
VAR o1,o2,o3
  F1_w1[1..2] -- weighted inputs
  F1_b1[1..2] -- biased inputs
  F1_th[1..2] -- activation functions
  F1_w2[1..2] -- weighted outputs
  F2_w1[1..2] -- weighted inputs
  F2_b1[1..2] -- biased inputs
  F2_th[1..2] -- activation functions
  F2_w2[1..2] -- weighted outputs
CONST
  F1_n0b[1..2] {val:[0.51134,
      -0.40334]}
  F1_n0w[1..2] {val:[-1.00454,
      -0.29309]}
  F1_n2b {val:0.26366}
  F1_n2w[1..2] {val:[0.40850,
      0.80369]}
  G1_a[1..2] {val:[1.00000,
      -0.87138]}
  G1_b[1..2] {val:[0.00000,
      -0.25726]}
  F2_n0b[1..2] {val:[0.27437,
      -0.25971]}
  F2_n0w[1..2] {val:[1.15853,
      -0.80192]}
  F2_n2b {val:0.03886}
  F2_n2w[1..2] {val:[-0.24270,
      0.67697]}
HISTORY o2 {dflt:0}
INIT
  F1_w1[1..2] := 0
  F1_b1[1..2] := 0
  F1_th[1..2] := 0
  F1_w2[1..2] := 0
  F2_w1[1..2] := 0
  F2_b1[1..2] := 0
  F2_th[1..2] := 0
  F2_w2[1..2] := 0
  o1 := 0
  o3 := 0
ENDINIT
EXEC
-- F1 block
  F1_w1[1..2] := i1 * F1_n0w[1..2]
  F1_b1[1..2] := F1_w1[1..2] + F1_n0b[1..2]
  F1_th[1..2] := tanh(F1_b1[1..2])
  FOR i:=1 to 2 DO F1_w2[i] := F1_th[i] * F1_n2w[i] ENDFOR
  o1 := F1_n2b
  FOR i:=1 to 2 DO o1 := o1 + F1_w2[i] ENDFOR
-- G1 block
  czfun(o2/o1) :=
 (G1_b[1]|z0+
      G1_b[2]|z-1)
  /
 (G1_a[1]|z0+
      G1_a[2]|z-1)
-- F2 block
  F2_w1[1..2] := o2 * F2_n0w[1..2]
  F2_b1[1..2] := F2_w1[1..2] + F2_n0b[1..2]
  F2_th[1..2] := tanh(F2_b1[1..2])
  FOR i:=1 to 2 DO F2_w2[i] := F2_th[i] * F2_n2w[i] ENDFOR
  o3 := F2_n2b
  FOR i:=1 to 2 DO o3 := o3 + F2_w2[i] ENDFOR
ENDEXEC
ENDMODEL
