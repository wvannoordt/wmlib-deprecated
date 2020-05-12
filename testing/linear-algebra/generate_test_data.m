clear
clc
close all
format long

rng(27891873);
ampl = 800000;
N = 500;

d1 = ampl*rand(N, 1);
d2 = ampl*rand(N-1, 1);
d3 = ampl*rand(N-1, 1);
r = ampl*rand(N, 1);

Mdiag = zeros(N,N);
for i = 1:N
    Mdiag(i,i) = d1(i);
end
for i = 1:N-1
    Mdiag(i,i+1) = d2(i);
    Mdiag(i+1,i) = d3(i);
end

xthom = Mdiag\r;
output_code_TDMA(Mdiag, r, xthom, 'thomas.hpp');


diag1 = ampl*rand(N, 1);
diag2 = ampl*rand(N, 1);
diag3 = ampl*rand(N, 1);

sub1 = ampl*rand(N-1, 1);
sub2 = ampl*rand(N-1, 1);
sub3 = ampl*rand(N-1, 1);

sup1 = ampl*rand(N-1, 1);
sup2 = ampl*rand(N-1, 1);
sup3 = ampl*rand(N-1, 1);

rhs = ampl*rand(2*N, 1);
M = zeros(2*N, 2*N);
for i = 1:N
    M(i,i) = diag1(i);
    M(i,i+N) = diag2(i);
    M(i+N,i+N) = diag3(i);
end
for i = 1:N-1
    M(i,i+1) = sup1(i);
    M(i,i+1+N) = sup2(i);
    M(i+N,i+1+N) = sup3(i);
    
    M(i+1,i) = sub1(i);
    M(i+1,i+N) = sub2(i);
    M(i+1+N,i+N) = sub3(i);
end

x = M\rhs;

output_code_TDMA_forward(M, rhs, x, 'tdma_forward.hpp');

