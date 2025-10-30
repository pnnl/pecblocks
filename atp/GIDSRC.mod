MODEL GIDSRC -- 6 character name limit
-- Start header. Do not modify the type-94 header. 
comment---------------------------------------------------------------
 | First, declarations required for any type 94 Norton non-TR model    |
 | - these data and input values are provided to the model by ATP      |
 | - these output values are used by ATP                               |
 | - these names can be changed, except 'n', but not their order       |
 -------------------------------------------------------------endcomment

DATA  n                      -- number of phases
      ng {dflt: n*(n+1)/2}   -- number of conductances on each side

INPUT v[1..n]   -- voltage(t) at each left node
      v0[1..n]  -- voltage(t=0) at each left node
      i0[1..n]  -- current(t=0) into each left node

VAR   i[1..n]   -- current(t) into each left node (for plotting)
      is[1..n]  -- Norton source(t+timestep) at each left node
      g[1..ng]  -- conductance(t+timestep) at each left node
                -- sequence is 1-gr, 1-2, 1-3..1-n,2-gr,2-3..2-n,...n-gr
      flag      -- set to 1 whenever conductance value is modified

OUTPUT i[1..n],is[1..n],g[1..ng],flag

 comment---------------------------------------------------------------
 | Next, declarations of user-defined data for this particular model   |
 | - their value is defined at the time of using the type-94 component |
 -------------------------------------------------------------endcomment
-- End header.  
DATA  WC {dflt: 377.0}
VAR o1,o2,o3,i1
  F1_w1[1..20] -- weighted inputs
  F1_b1[1..20] -- biased inputs
  F1_th[1..20] -- activation functions
  F1_w2[1..20] -- weighted outputs
  F2_w1[1..20] -- weighted inputs
  F2_b1[1..20] -- biased inputs
  F2_th[1..20] -- activation functions
  F2_w2[1..20] -- weighted outputs
CONST
  F1_n0b[1..20] {val:[0.54338,
      0.44840,
      0.57686,
      0.26182,
      0.72317,
      -0.64570,
      -0.11028,
      0.03978,
      -0.34981,
      0.46116,
      0.52261,
      -0.48527,
      -0.35774,
      -0.28330,
      0.67468,
      -0.73765,
      0.75258,
      -0.43485,
      -0.23745,
      -0.21229]}
  F1_n0w[1..20] {val:[-0.79690,
      0.03021,
      0.99701,
      -0.75577,
      0.05870,
      0.49827,
      0.26999,
      -0.84319,
      0.25172,
      0.56409,
      -0.58113,
      0.65927,
      -0.52601,
      -0.13836,
      0.55831,
      0.11199,
      -0.88813,
      0.63315,
      0.69850,
      0.63784]}
  F1_n2b {val:0.18069}
  F1_n2w[1..20] {val:[-0.27112,
      0.29404,
      0.20127,
      -0.42489,
      -0.04443,
      0.16317,
      0.35116,
      -0.18968,
      0.16325,
      0.07260,
      -0.16228,
      0.27121,
      -0.23350,
      -0.27648,
      0.19077,
      0.08621,
      -0.27675,
      0.19209,
      0.42542,
      0.26594]}
  G1_a[1..2] {val:[1.00000,
      -0.98619]}
  G1_b[1..2] {val:[0.00000,
      0.13876]}
  F2_n0b[1..20] {val:[-0.11055,
      0.86216,
      0.92846,
      0.42168,
      -0.05532,
      -1.00675,
      0.08724,
      -0.41628,
      0.10016,
      -0.45086,
      -0.14550,
      0.61503,
      1.30816,
      0.81967,
      -0.57617,
      0.64028,
      0.57698,
      0.42654,
      1.48592,
      0.05310]}
  F2_n0w[1..20] {val:[-3.17689,
      0.52152,
      0.69806,
      -1.55514,
      0.49624,
      -0.18063,
      0.55527,
      1.66691,
      -0.70190,
      0.30797,
      0.47506,
      0.05924,
      -0.70314,
      -0.37020,
      -0.28872,
      0.74010,
      0.97298,
      -0.30992,
      -0.75922,
      -0.41204]}
  F2_n2b {val:-0.05206}
  F2_n2w[1..20] {val:[-0.08112,
      -0.13599,
      -0.05778,
      0.09454,
      -0.10788,
      0.07061,
      0.25891,
      -0.19385,
      -0.02241,
      -0.03505,
      0.24303,
      -0.12252,
      0.23065,
      0.07737,
      0.22813,
      0.02912,
      0.03574,
      -0.23898,
      0.11427,
      0.08721]}
HISTORY o2 {dflt:0}
INIT
  F1_w1[1..20] := 0
  F1_b1[1..20] := 0
  F1_th[1..20] := 0
  F1_w2[1..20] := 0
  F2_w1[1..20] := 0
  F2_b1[1..20] := 0
  F2_th[1..20] := 0
  F2_w2[1..20] := 0
  o1 := 0
  o3 := 0
-- conductance
  g[1] := 0.000001  -- small shunt on the output terminal
  g[2] := 0.0       -- no coupling from output to G input
  g[3] := 0.0       -- no loading of the G input
ENDINIT
EXEC
-- initialize conductance
  if t=0 then
    flag:=1
  else
    flag:=0
  endif
-- F1 block
  i1 := v[2]
  F1_w1[1..20] := i1 * F1_n0w[1..20]
  F1_b1[1..20] := F1_w1[1..20] + F1_n0b[1..20]
  F1_th[1..20] := tanh(F1_b1[1..20])
  FOR i:=1 to 20 DO F1_w2[i] := F1_th[i] * F1_n2w[i] ENDFOR
  o1 := F1_n2b
  FOR i:=1 to 20 DO o1 := o1 + F1_w2[i] ENDFOR
-- G1 block
  czfun(o2/o1) :=
 (G1_b[1]|z0+
      G1_b[2]|z-1)
  /
 (G1_a[1]|z0+
      G1_a[2]|z-1)
-- F2 block
  F2_w1[1..20] := o2 * F2_n0w[1..20]
  F2_b1[1..20] := F2_w1[1..20] + F2_n0b[1..20]
  F2_th[1..20] := tanh(F2_b1[1..20])
  FOR i:=1 to 20 DO F2_w2[i] := F2_th[i] * F2_n2w[i] ENDFOR
  o3 := F2_n2b
  FOR i:=1 to 20 DO o3 := o3 + F2_w2[i] ENDFOR
-- inject the Norton sources
  i[1] := o3
  is[1] := o3
ENDEXEC
ENDMODEL
