OPENQASM 2.0;
include "qelib1.inc";
gate cDiagonal q0,q1,q2 { p(1*pi/8) q0; p(1*pi/8) q1; cx q0,q1; u(0,0,-1*pi/8) q1; cx q0,q1; u(0,0,0) q1; ccx q0,q2,q1; p(0.39126866) q0; p(0.39126866) q1; cx q0,q1; u(0,0,-0.39126866) q1; cx q0,q1; u(0,0,0) q1; ccx q0,q2,q1; p(-1.1766655) q0; p(-1.1766655) q2; cx q0,q2; u(0,0,1.1766655) q2; cx q0,q2; u(0,0,0) q2; p(9*pi/2) q0; }
gate c_Diagonal q0,q1,q2 { cDiagonal q0,q1,q2; }
gate gate__QFT__dagger__dg q0,q1 { h q0; cp(-pi/2) q0,q1; h q1; }
gate c_Diagonal__1 q0,q1,q2 { c_Diagonal q0,q1,q2; }
gate gate__QFT__dagger__dg_140414069380912 q0 { h q0; }
creg result[1];
qreg qregless_0[6];
x qregless_0[3];
x qregless_0[2];
x qregless_0[0];
h qregless_0[4];
h qregless_0[1];
cp(-0.1728465) qregless_0[3],qregless_0[1];
cp(-0.345693) qregless_0[3],qregless_0[4];
cp(-0.28529588) qregless_0[2],qregless_0[1];
cp(1.974491) qregless_0[0],qregless_0[1];
cp(-0.57059177) qregless_0[2],qregless_0[4];
cp(3.948982) qregless_0[0],qregless_0[4];
gate__QFT__dagger__dg qregless_0[4],qregless_0[1];
h qregless_0[5];
c_Diagonal__1 qregless_0[5],qregless_0[4],qregless_0[1];
gate__QFT__dagger__dg_140414069380912 qregless_0[5];
measure qregless_0[5] -> result[0];
