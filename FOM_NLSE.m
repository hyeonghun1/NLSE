clc;
clear all;
% This code was used for the work Sharma, Wang, Kramer, 2022, Physica D
% 431,especially the implementation of the FOM 1D NLSE

%% Problem set-up
% Case 1
T       = 100;           % final time
L       = 2*sqrt(2)*pi;
dx      = L/64;          % mesh width
dt      = 0.005;         % time step
x_grid  = -L/2:dx:L/2;   % grid points
t       = 0:dt:T;        % time points
Nt      = T/dt;
gam     = 2;

N = length(x_grid)-1;
X = zeros(2*N, Nt);

% Initial conditions
psi0 = init(x_grid, N, L);
psi = psi0;

% IC snapshot
X(:,1) = psi;

% time stepping
for i=1:Nt
    % Initial guess for (k+1)th iteration based on the (k)th and (k-1)th iteration values
    psi_next = psi;
    
    % Setting up counter and error variables for solving the system of
    % 2N nonlinear equations
    ci=0;
    err=1;
    while err > 1e-13
        [fvec, J] = NLSE_solve(psi_next, psi, dx, dt, gam, N);
        %
        psi_next = psi_next - (J\fvec);
        ci = ci + 1;
        err = norm(fvec);
        if ci==10
            err
            err=0.5e-14;
        end
    end
    % ci;
    X(:,i+1) = psi_next;
    psi = psi_next;
end
% %
P   = X(1:N, :);         % p snapshots
Q   = X(N+1:end, :);     % q snapshots
Psi = abs(P + Q*1j);     % |psi| snapshots
% 

% % % Plotting the solution and energy
figure;
surf(x_grid(2:end)', t', (Psi(1:N,:).^2)','LineStyle','none')
title('Deterministic 1D NLSE solution', FontSize=20);
xlabel('$x$', Interpreter='latex', FontSize=20);
ylabel('$t$', Interpreter='latex', FontSize=20);
zlabel('$|\psi|^2$', Interpreter='latex', FontSize=20);
colormap jet; colorbar;

% % Hamiltonian, mass/momentum invariants
[H, M1, M2, H2] = ener(Q, P, N, Nt, gam, dx);
% H2 = enermat(Q, P, N, Nt, c, dx);

figure;
plot(t, H, 'LineWidth', 2); hold on;
plot(t, H-H2, 'LineWidth', 2, 'LineStyle', '-.');
plot(t, H2, 'LineWidth', 2, 'LineStyle', ':');
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('Energy', 'FontSize', 20);
legend('Total energy', 'Kinetic only', 'Potential only', fontsize=16)
set(gca, 'Fontsize', 12);
%

figure;
subplot(1,2,1); plot(t, M1, 'LineWidth', 2);
title('Mass invariant', fontsize=20);
xlabel('t', 'FontSize', 20);
set(gca, 'Fontsize', 20);
subplot(1,2,2); plot(t, M2, 'LineWidth', 2);
title('Momentum invariant', fontsize=20);
xlabel('t', 'FontSize', 20);
set(gca, 'Fontsize', 20);

%% Solve this same 1D NLSE using split-stepping (operator splitting)

% Spatial parameters
Nx = 2^6;                     % number of spatial grid points
Lx = 2 * sqrt(2) * pi;        % domain length
dx = Lx / Nx;                 % spatial step
x  = linspace(-Lx/2, Lx/2 - dx, Nx)';  % periodic grid (endpoint excluded)

% Temporal parameters
dt = 0.005;
Nt = 20000;
t  = (0:Nt) * dt;

% NLSE parameters
kappa = 2.0;

% Initial condition
A = 0.5;
psi0 = A * (1 + 0.01 * cos(2*pi*x/Lx));   % initial wave
psi  = psi0;                              % complex field

% Wavenumbers (FFT)
k = 2*pi * [0:Nx/2-1 -Nx/2:-1]' / Lx;
k2 = k.^2;

% Linear propagator
L = exp(-1i * k2 * dt);   % exact linear evolution operator

% Storage for results
snapshots = zeros(Nx, floor(Nt/100)+1);
snap_idx = 1;
snapshots(:, snap_idx) = abs(psi).^2;
snap_idx = snap_idx + 1;

% Time stepping using Split-Step Fourier Method 
for n = 1:Nt
    % Nonlinear half-step
    psi = psi .* exp(1i * kappa * abs(psi).^2 * dt/2);
    
    % Linear full-step
    psi_hat = fft(psi);
    psi_hat = psi_hat .* L;
    psi = ifft(psi_hat);
    
    % Nonlinear half-step
    psi = psi .* exp(1i * kappa * abs(psi).^2 * dt/2);
    
    snapshots(:, n) = abs(psi).^2;
    % snap_idx = snap_idx + 1;
end

% Plot snapshots
plot_every = 100;
figure;
hold on;
for i = 1:plot_every:size(snapshots, 2)
    plot(x, snapshots(:, i), 'LineWidth', 1.2);
end
xlabel('$x$', Interpreter='latex', FontSize=20);
ylabel('$|\psi|^2$', Interpreter='latex', FontSize=20);
title(['1D NLSE Split-Step Fourier Method, \gamma = ', num2str(kappa)], FontSize=20);
grid on; hold off;

% 3D surface plot
t_full = linspace(0, Nt*dt, size(snapshots, 2));  % full time array
[T, X] = meshgrid(t_full, x);

figure;
surf(X, T, snapshots, 'EdgeColor', 'none');
xlabel('$x$', Interpreter='latex', FontSize=20);
ylabel('$t$', Interpreter='latex', FontSize=20);
zlabel('$|\psi|^2$', Interpreter='latex', FontSize=20);
colormap jet; colorbar;
title(['1D NLSE evolution, \gamma = ', num2str(kappa)]);
% view(45,30);
shading interp;

%% Optional animation
% figure;
% hLine = plot(x, abs(psi0).^2, 'b');
% xlabel('x'); ylabel('$|\psi|^2$', Interpreter='latex');
% title('Deterministic 1D NLSE Evolution');
% ylim([0, max(abs(psi0).^2)*1.5]);
% grid on;
% 
% for n = 1:size(snapshots,2)
%     set(hLine, 'YData', snapshots(:,n));
%     title(['t = ', num2str((n-1)*plot_every*dt, '%.3f')]);
%     drawnow;
% end

%% --- Save all variables ---
% save('NLSE_deterministic_SSFM.mat');


%% Function for defining the initial conditions for NLSE problem
function [y0] = init(x_grid, N, L)
q0 = zeros(N,1);  % imag part
p0 = zeros(N,1);  % real part 
for j=1:1:N
    p0(j,1) = 0.5*(1+ (0.01*cos(2*pi*x_grid(j)/L)));
    q0(j,1) = 0;
end
y0=[p0; q0;];
end
%% Function for building matrix for discrete second order differential operator
function [A] = Dxx(N)
A=-2*eye(N);
A(1,2)=1;
A(1,N)=1;
A(N,1)=1;
A(N,N-1)=1;
for i=2:1:N-1
   A(i,i-1)=1;
   A(i,i+1)=1;
end
end
%% Function for matrices for FOM Hamiltonian
function [Ap, Aq] = Hmat(c, N, dx)
Aq=4*eye(N);
Aq(1,2)=-2;
Aq(1,N)=-2;
Aq(N,1)=-2;
Aq(N,N-1)=-2;
for i=2:1:N-1
   Aq(i,i-1)=-2;
   Aq(i,i+1)=-2;
end
Aq=(c*c*Aq)/(4*dx);
Ap=dx*eye(N)/2;
end
%% Function for computing the total energy H, mass invariant M1 and momentum invariant M2 
function [H, M1, M2, H2] = ener(Q, P, N, Nt, gam, dx)
H=zeros(Nt+1,1);
H2=zeros(Nt+1,1);
M1=zeros(Nt+1,1);
M2=zeros(Nt+1,1);
for i=1:Nt+1
    for j=1:N-1
    H(i,1)=H(i,1) + 0.5*((P(j+1,i)-P(j,i))/dx)^2 + 0.5*((Q(j+1,i)-Q(j,i))/dx)^2 - 0.25*gam*((Q(j,i)^2 + P(j,i)^2)^2);
    H2(i,1)=H2(i,1) - 0.25*gam*((Q(j,i)^2 + P(j,i)^2)^2);
    M1(i,1)=M1(i,1) + 0.5*((P(j,i))^2) + 0.5*((Q(j,i))^2);
    M2(i,1)=M2(i,1) + ( (P(j+1,i)-P(j,i)) * Q(j,i) ) -  ( (Q(j+1,i)-Q(j,i)) * P(j,i) );
    end
     H(i,1)=H(i,1) + 0.5*((P(1,i)-P(N,i))/dx)^2 + 0.5*((Q(1,i)-Q(N,i))/dx)^2 - 0.25*gam*((Q(N,i)^2 + P(N,i)^2)^2);
     H2(i,1)=H2(i,1)  - 0.25*gam*((Q(N,i)^2 + P(N,i)^2)^2);
     M1(i,1)=M1(i,1) + 0.5*((P(N,i))^2) + 0.5*((Q(N,i))^2);
     M2(i,1)=M2(i,1) + ( (P(1,i)-P(N,i)) * Q(N,i) ) -  ( (Q(1,i)-Q(N,i)) * P(N,i) );
    
end
H=H*dx;
H2=H2*dx;
M1=M1*dx;

end
%% Function for solving the system of nonlinear equations
% function [fvec, J] = NLSE_solve(y_next, y_now, dx, dt, gam, N)
% % First point case
%     % Use periodic BC
%     ak = [y_now(N, 1); y_now(1:2, 1); y_now(2*N, 1); y_now(N+1:N+2, 1);];
%     ak1= [y_next(N, 1); y_next(1:2, 1); y_next(2*N, 1); y_next(N+1:N+2, 1);];
%     %
%     [residual, jac] = NLSE_eq(ak1, ak, dx, dt, gam);
%     %
%     fvec(1,1)   = residual(1,1);
%     fvec(N+1,1) = residual(2,1);
%     %
%     J(1,N)         = jac(1,1);
%     J(1,1:2)       = jac(1,2:3);
%     J(1,2*N)       = jac(1,4);
%     J(1,N+1:N+2)   = jac(1,5:6);
%     J(N+1,N)       = jac(2,1);
%     J(N+1,1:2)     = jac(2,2:3);
%     J(N+1,2*N)     = jac(2,4);
%     J(N+1,N+1:N+2) = jac(2,5:6);
% %
% for i=2:N-1
%     ak=[y_now(i-1:i+1,1); y_now(N+i-1:N+i+1,1)];
%     ak1=[y_next(i-1:i+1,1); y_next(N+i-1:N+i+1,1)];
%     %
%     [residual, jac] = NLSE_eq(ak1, ak, dx, dt, gam);
%     %
%     fvec(i,1)   = residual(1,1);
%     fvec(N+i,1) = residual(2,1);
%     %
%     J(i,i-1:i+1)       = jac(1,1:3);
%     J(i,N+i-1:N+i+1)   = jac(1,4:6);
%     J(N+i,i-1:i+1)     = jac(2,1:3);
%     J(N+i,N+i-1:N+i+1) = jac(2,4:6);
% end
% %
% % last point case
%     ak=[y_now(N-1:N,1);y_now(1,1);y_now(2*N-1:2*N,1);y_now(N+1,1);];
%     ak1=[y_next(N-1:N,1);y_next(1,1);y_next(2*N-1:2*N,1);y_next(N+1,1);];
%     %
%     [residual, jac] = NLSE_eq(ak1, ak, dx, dt, gam);
%     %
%     fvec(N,1)   = residual(1,1);
%     fvec(2*N,1) = residual(2,1);
%     %
%     J(N,N-1:N)       = jac(1,1:2);
%     J(N,1)           = jac(1,3);
%     J(N,2*N-1:2*N)   = jac(1,4:5);
%     J(N,N+1)         = jac(1,6);
%     J(2*N,N-1:N)     = jac(2,1:2);
%     J(2*N,1)         = jac(2,3);
%     J(2*N,2*N-1:2*N) = jac(2,4:5);
%     J(2*N,N+1)       = jac(2,6);
% %
% end
% %%
% function [residual, jac] = NLSE_eq(y_next, y_now, dx, dt, gam)
% %
% q1ik1 = y_next(1,1);    % p^{k+1}_{i-1}
% qik1 = y_next(2,1);     % p^{k+1}_i
% qi1k1 = y_next(3,1);    % p^{k+1}_{i+1}
% p1ik1 = y_next(4,1);    % q^{k+1}_{i-1}
% pik1 = y_next(5,1);     % q^{k+1}_i
% pi1k1 = y_next(6,1);    % q^{k+1}_{i+1}
% %
% q1ik = y_now(1,1);
% qik = y_now(2,1);
% qi1k = y_now(3,1);
% p1ik = y_now(4,1);
% pik = y_now(5,1);
% pi1k = y_now(6,1);
% %
% residual = [qik1 - qik + (dt*(p1ik/2 + p1ik1/2 + pi1k/2 + pi1k1/2 - pik - pik1)/dx^2) + (dt*gam*((pik/2 + pik1/2)^2 + (qik/2 + qik1/2)^2)*(pik/2 + pik1/2));
% pik1 - pik - (dt*(q1ik/2 + q1ik1/2 + qi1k/2 + qi1k1/2 - qik - qik1)/dx^2) - (dt*gam*(qik/2 + qik1/2)*((pik/2 + pik1/2)^2 + (qik/2 + qik1/2)^2));];
% %
% jac =[     0, dt*gam*(qik/2 + qik1/2)*(pik/2 + pik1/2) + 1,                                                          0, dt/(2*dx^2), (dt*gam*((pik/2 + pik1/2)^2 + (qik/2 + qik1/2)^2))/2 - dt/dx^2 + dt*gam*(pik/2 + pik1/2)^2, dt/(2*dx^2);
% -dt/(2*dx^2), dt/dx^2 - (dt*gam*((pik/2 + pik1/2)^2 + (qik/2 + qik1/2)^2))/2 - dt*gam*(qik/2 + qik1/2)^2, -dt/(2*dx^2),           0,                                               1 - dt*gam*(qik/2 + qik1/2)*(pik/2 + pik1/2),           0];
% %
% 
% % disp(size(residual))
% end

%% Efficient Newton's iteration
function [fvec, J] = NLSE_solve(y_next, y_now, dx, dt, gam, N)
% Efficient residual and Jacobian for 1D NLSE implicit scheme
%
% Inputs:
% y_now, y_next : [2N x 1] real vectors [p; q]
% dx, dt, gam   : discretization, nonlinearity parameters
% N             : number of spatial grid points
%
% Output:
% fvec [2N x 1]  : nonlinear residual
% J    [2N x 2N] : Jacobian matrix

% ----------------------------------------
% Preallocate residual and Jacobian arrays
% ----------------------------------------
fvec = zeros(2*N, 1);
% sparse preallocation (each row ~6 nonzeros)
J = spalloc(2*N, 2*N, 12*N);   

% ------------------------------
% Split state into q and p parts
% ------------------------------
p_now  = y_now(1:N);
q_now  = y_now(N+1:end);
p_next = y_next(1:N);
q_next = y_next(N+1:end);

% -----------------------------------------------------
% Periodic extension (to avoid special boundary cases)
% -----------------------------------------------------
p_now_ext  = [p_now(end); p_now; p_now(1)];
q_now_ext  = [q_now(end); q_now; q_now(1)];
p_next_ext = [p_next(end); p_next; p_next(1)];
q_next_ext = [q_next(end); q_next; q_next(1)];

% -----------------------------------------------------
% Loop through all interior points (uniform treatment)
% -----------------------------------------------------
for i = 1:N
    % Local 3-point stencil indices (periodic wrap)
    idx = i + (0:2);  % maps to i-1, i, i+1 in extended arrays

    % Collect local triplets
    pk  = p_now_ext(idx);
    qk  = q_now_ext(idx);
    pk1 = p_next_ext(idx);
    qk1 = q_next_ext(idx);

    % Local nonlinear residual and Jacobian (2x6)
    [r, jac] = NLSE_eq_local(pk1, qk1, pk, qk, dx, dt, gam);

    % Fill global residual
    fvec(i)   = r(1);
    fvec(N+i) = r(2);

    % Indices for q and p in global system
    i_prev = mod(i-2, N) + 1; % i-1
    i_curr = i;               % i
    i_next = mod(i, N) + 1;   % i+1
    
    i_fill = [i_prev, i_curr, i_next, ...
                N+i_prev, N+i_curr, N+i_next];

    % Place local Jacobian into sparse global matrix
    J(i,   i_fill) = jac(1,:);
    J(N+i, i_fill) = jac(2,:);
end
end


function [residual, jac] = NLSE_eq_local(pk1, qk1, pk, qk, dx, dt, gam)
% Local 3-point Crank–Nicolson (midpoint rule) for NLSE (real form)
%
% Inputs:
% qk1, pk1 : 3-point vectors at time n+1
% qk,  pk  : 3-point vectors at time n
%
% Outputs:
% residual : local residual vector (6 x 1) 
% jac      : local Jacobian matrix (2 x 6)
% --------------------------------------------

% Midpoint nonlinear term (Crank–Nicolson)
pmid = 0.5 * (pk(2) + pk1(2)); % midpoint over time at i
qmid = 0.5 * (qk(2) + qk1(2));
r2mid = qmid^2 + pmid^2;

% Laplacians (central difference)
dp_now  = (pk(1) - 2*pk(2) + pk(3)) / dx^2;
dq_now  = (qk(1) - 2*qk(2) + qk(3)) / dx^2;
dp_next = (pk1(1) - 2*pk1(2) + pk1(3)) / dx^2;
dq_next = (qk1(1) - 2*qk1(2) + qk1(3)) / dx^2;

% Crank–Nicolson (average Laplacian)
dp_avg = 0.5 * (dp_now + dp_next);
dq_avg = 0.5 * (dq_now + dq_next);

% --------------------------------------------
% Residuals (real and imaginary parts)
% --------------------------------------------
residual = [
    pk1(2) - pk(2) + dt * ( dq_avg + gam * r2mid * qmid );  % p-equation
    qk1(2) - qk(2) - dt * ( dp_avg + gam * r2mid * pmid )   % q-equation
];

% --------------------------------------------
% Analytical Jacobian (2x6)
% --------------------------------------------
deriv_Lap = [1, -2, 1] / (2*dx^2); % average of Laplacians over k, k+1

% partials of nonlinear term
% dr2_dpk1 = [0, pmid, 0]; 
% dr2_dqk1 = [0, qmid, 0];

% Initialization
jac = zeros(2,6);

% Row 1: derivative of p-residual
% dq_avg term → acts on q^{k+1}
jac(1,4:6) = dt * deriv_Lap;  
% nonlinear term wrt p^{k+1}
jac(1,2) = jac(1,2) + dt*gam*(2*pmid*qmid)/2;
% nonlinear term wrt q^{k+1}
jac(1,5) = jac(1,5) + dt*gam*(3*qmid^2 + pmid^2)/2;
% identity term
jac(1,2) = jac(1,2) + 1;

% Row 2: derivative of q-residual
% dp_avg term → acts on p^{n+1}
jac(2,1:3) = -dt * deriv_Lap;
% nonlinear term wrt p^{n+1}
jac(2,2) = jac(2,2) - dt*gam*(3*pmid^2 + qmid^2)/2;
% nonlinear term wrt q^{n+1}
jac(2,5) = jac(2,5) - dt*gam*(2*pmid*qmid)/2;
% identity term
jac(2,5) = jac(2,5) + 1;

end