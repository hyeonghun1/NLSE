clc; clear all;
% This code lets you practice simulating Stochastic Differential Equations
% (SDEs) numerically. The entire code is from Desmond J. Higham, 2001, SIAM
% Review, Vol. 43, No. 3, 525-546.

%%
% BPATH1 Brownian path simulation
% This part generates discretized Brownian path

randn('state', 100 ) % for reproducibility

T = 1;
N = 500;
dt = T/N;

dW = zeros(1,N); % preallocate arrays
W  = zeros(1,N);  % for efficiency

dW(1) = sqrt(dt)*randn; % first increment of the Brownian motion
W(1) = dW(1); % since W(0) = 0 is not allowed (we set W(0)=0 when plotting)

for j = 2:N
    dW(j) = sqrt(dt)*randn;  % general increments of W(t)
    W(j) = W(j-1) + dW(j);
end

figure;
plot([0:dt:T], [0, W], 'r-') % plot W against t
xlabel('t','FontSize', 16)
ylabel('W(t)', 'FontSize', 16, 'Rotation', 0)
title('A discretized Brownian path', 'FontSize', 16)


%%
% BPATH2 Brownian path simulation: vectorized

randn('state', 100) % set the state of randn
T = 1;
N = 500;
dt = T/N;

% increments of Bwonian motion ~ \sqrt(dt) * N(0,1)
dW = sqrt(dt) * randn(1, N); 
W = cumsum(dW); % cumulative sum - this helps avoid for loops

figure;
plot([0:dt:T], [0, W], 'r-') % plot W against t
xlabel('t', 'FontSize',16)
ylabel('W(t)', 'FontSize', 16, 'Rotation', 0)
title('A discretized Brownian path', 'FontSize', 16)


%%
% BPATH3 Function along a Brownian path
randn('state', 100) % set the state of randn
T = 1;
N = 500;
dt = T/N;
t = [dt:dt:1];

M  = 1000; % M paths simultaneously
dW = sqrt(dt) * randn(M,N); % increments
W  = cumsum(dW, 2); % cumulative sum: 1000 realizations of W(t) over 500 time instances

% Evaluate the function u(W(t)) = exp(t + 0.5W(t))
U  = exp(repmat(t, [M 1]) + 0.5*W); 
Umean = mean(U);        % columnwise average

figure;
plot([0,t], [1, Umean], 'b-'), hold on % plot mean over M paths
plot([0,t], [ones(5,1), U(1:5,:)],'r--'), hold off % plot 5 individual paths
xlabel('t', 'FontSize',16)
ylabel('U(t)',  'FontSize', 16, 'Rotation', 0, 'HorizontalAlignment', 'right')
legend('mean of 1000 paths', '5 individual paths', 'FontSize', 16)

% The sample average (expected value of u(W(t)) turns out to be exp(9t/8)
averr = norm((Umean - exp(9*t/8)), 'inf')   % sample error -> this will decrease when sampled more W(t)


%%
%STINT Approximate stochastic integrals
%
% Ito and Stratonovich integrals of W dW

randn('state', 100) % set the state of randn
T = 1;
N = 500;
dt = T/N;

dW = sqrt(dt) * randn(1,N); % increments
W = cumsum(dW);             % cumulative sum

ito = sum([0, W(1:end-1)] .* dW)
strat = sum((0.5*([0,W(1:end-1)]+W) + 0.5*sqrt(dt)*randn(1,N)).*dW)

itoerr = abs(ito - 0.5*(W(end)^2 - T))
straterr = abs(strat - 0.5*W(end)^2)


%% EM Euler-Maruyama method on linear SDE
%
% SDE is dX = lambda*X dt + mu*X dW, X(0) = Xzero,
% where lambda = 2, mu = 1 and Xzero = 1.
%
% Discretized Brownian path over [0,1] has dt = 2^(-8).
% Euler-Maruyama uses timestep R*dt.
randn('state', 100)
lambda = 2;
mu = 1;
Xzero = 1; % problem parameters

T = 1;
N = 2^8;
dt = 1/N;

dW = sqrt(dt)*randn(1,N); % Brownian increments
W = cumsum(dW);           % discretized Brownian path

% Exact solution
Xtrue = Xzero * exp((lambda-0.5*mu^2) * ([dt:dt:T]) + mu*W);

figure;
plot([0:dt:T], [Xzero, Xtrue], 'm-'), hold on

R = 4;
Dt = R*dt;  % step size of EM
L = N/R;    % L EM steps of size Dt = R*dt
Xem = zeros(1,L); % preallocate for efficiency
Xtemp = Xzero;

for j = 1:L
    Winc = sum(dW(R*(j-1) + 1:R*j));    % W increment in EM
    Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;
    Xem(j) = Xtemp;
end

plot([0:Dt:T], [Xzero, Xem],  'r--*'), hold off
xlabel('t', 'FontSize', 12)
ylabel('X', 'FontSize', 16, 'Rotation', 0, 'HorizontalAlignment', 'right')
legend('exact solution', 'numerical solution', 'FontSize', 16)

% Error at the end point
emerr = abs(Xem(end) - Xtrue(end))   % this will decrease fior smaller R value



%% EMSTRONG Test strong convergence of Euler-Maruyama
%
% Solves dX = lambda*X dt + mu*X dW, X(0) = Xzero,
% where lambda = 2, mu = 1 and Xzer0 = 1.
%
% Discretized Brownian path over [0,1] has dt = 2^(-9).
% E-M uses 5different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
% Examine strong convergence at T=1: E | X_L - X(T) |.

randn('state', 100)
lambda = 2; mu = 1; Xzero = 1; % problem parameters
T = 1;
N = 2^9;
dt = T/N;

M = 1000;               % number of paths sampled
Xerr = zeros(M,5);      % preallocate array

% Sample over discrete Brownian paths
for s = 1:M 
    dW = sqrt(dt)*randn(1,N);   % Brownian increments
    W = cumsum(dW);             % discrete Brownian path
    Xtrue = Xzero*exp((lambda-0.5*mu^2)+mu*W(end));
    
    % Test with 5 different step sizes
    for p = 1:5
        R = 2^(p-1);
        Dt = R*dt;
        L = N/R; % L Euler steps of size Dt = R*dt
        Xtemp = Xzero;
        
        for j = 1:L
            Winc = sum(dW(R*(j-1)+1:R*j));
            Xtemp = Xtemp + Dt*lambda*Xtemp + mu*Xtemp*Winc;
        end
    
        Xerr(s,p) = abs(Xtemp - Xtrue); % store the error at t = 1
    end
end

Dtvals = dt*(2.^([0:4]));

figure;
subplot(221) % top LH picture
loglog(Dtvals, mean(Xerr), 'b*-'), hold on
loglog(Dtvals, (Dtvals.^(.5)), 'r--'), hold off % reference slope of 1/2
axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t'), ylabel('Sample average of | X(T) - X_L |')
title('emstrong.m', 'FontSize', 10)

%%%% Least squares fit of error = C * Dt^q %%%%
A = [ones(5,1), log(Dtvals)'];
rhs = log(mean(Xerr)');

sol = A\rhs;
q = sol(2)
resid = norm(A*sol - rhs)


%% CHAIN Test stochastic Chain Rule
%
% Solve SDE for V(X) = sqrt(X) where X solves
% dX = (alpha - X) dt + beta sqrt(X) dW, X(0) = Xzero,
% with alpha = 2, beta = 1 and Xzero = 1.
% Xem1 is Euler-Maruyama solution for X.
% Xem2 is Euler-Maruyama solution of SDE for V from Chain Rule.
% Hence, we compare sqrt(Xem1) and Xem2.
% Note: abs is used for safety inside sqrt, but has no effect in this case.

randn('state', 100)
alpha = 2;
beta = 1;
T = 1;
N = 200;
dt = T/N; % Problem parameters

Xzero = 1;
Xzero2 = 1/sqrt(Xzero);

Dt = dt; % EM steps of size Dt = dt
Xem1 = zeros(1,N);
Xem2 = zeros(1,N); % preallocate for efficiency
Xtemp1 = Xzero;
Xtemp2 = Xzero2;

for j = 1:N
    Winc = sqrt(dt)*randn;
    f1 = (alpha-Xtemp1);
    g1 = beta*sqrt(abs(Xtemp1));

    Xtemp1 = Xtemp1 + Dt*f1 + Winc*g1;
    Xem1(j) = Xtemp1;
    
    f2 = (4*alpha-beta^2)/(8*Xtemp2) - Xtemp2/2;
    g2 = beta/2;
    Xtemp2 = Xtemp2 + Dt*f2 + Winc*g2;
    Xem2(j) = Xtemp2;
end

figure;
plot([0:Dt:T],[sqrt([Xzero,abs(Xem1)])],  'b-', [0:Dt:T], [Xzero,Xem2], 'ro')
legend('Direct Solution',  'Solution via Chain Rule', 'FontSize', 16)
xlabel('t', 'FontSize', 12)
ylabel('V(X)',  'FontSize', 16, 'Rotation', 0, 'HorizontalAlignment', 'right')

Xdiff = norm(sqrt(Xem1) - Xem2, 'inf')


