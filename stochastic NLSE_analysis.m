clc; clear;
% This code simulates 1D stochastic NLSE using several numerical schemes,
% such as explicit, implicit (backward or midpoint) Euler-Maruyama schemes.

%% Implicit (Crank-Nicolson) + Newton's method

T   = 100;           % final time
dt  = 0.001;           % time step

L   = 2*sqrt(2)*pi;  % domian length
N   = 64;            % number of spatial points
dx  = L/N;           % mesh width
dt  = 0.005;         % time step
x   = linspace(-L/2, L/2-dx, N)';  % spatial grid
t   = 0:dt:T;        % time points
Nt  = T/dt;
gam = 2;


% Initial condition
p0 = 0.5*(1 + 0.01*cos(2*pi*x/L));
q0 = zeros(N,1);

% Preallocate solution
P = zeros(N, Nt+1);
P(:,1) = p0;
Q = zeros(N, Nt+1);
Q(:,1) = q0;

X = zeros(2*N, Nt);

% Initial conditions
psi0 = init(x_grid, N, L);
psi = psi0;

% IC snapshot
X(:,1) = psi;

% Noise amplitude - if this is very small, the solution looks similar to
% the deterministic case, but it too large the solution loses physical
% meaning (the solution will not look like solitons)
sigma = 0.01; 

% rng('default');  % for reproducibility

BC_type = 'periodic';   % periodic, dirichlet, neumann          
for i = 1:Nt
    
    % Initial guess for Newton
    psi_next = psi; 

    if strcmpi(BC_type, 'dirichlet')
        t_n = t(i);
        [pL, pR, qL, qR] = BCfunction(t_n, T);
    end

    % Sample additive noise (complex)
    xi = randn(2*N, 1); % independent for p and q parts
    noise = sigma * sqrt(dt) * xi;

    % Newton iteration
    ci = 0;
    err = 1;
    
    while err > 1e-13

    % Force BCs at current iteration
    if strcmpi(BC_type, 'dirichlet')
        psi_next(1)   = pL;     % Set boundaries for psi_next
        psi_next(N)   = pR;
        psi_next(N+1) = qL;
        psi_next(N+N) = qR;
    end
    
    % [fvec, J] = NLSE_solve(psi_next, psi, dx, dt, gam, N);
    [fvec, J] = NLSE_solve_BC(psi_next, psi, dx, dt, gam, N, BC_type);
    % [fvec, J] = NLSE_solve_BC(psi_next, psi, dx, dt, gam, N, BC_type, pL, pR, qL, qR);

    
    % Include noise to residual (only to fvec, not to Jacobian)
    fvec = fvec - noise;
    
    % Update Newton
    psi_next = psi_next - (J \ fvec);
    ci = ci + 1;
    err = norm(fvec);
    if ci == 10
        err
        err = 0.5e-14;
    end

    end

    % Store solution
    X(:,i+1) = psi_next;
    psi = psi_next;
end

% %
P   = X(1:N, :);         % p snapshots
Q   = X(N+1:end, :);     % q snapshots
Psi = abs(P + Q*1j);     % |psi| snapshots
% 

% % Plotting the solution (mass density/power)
figure;
surf(x_grid(2:end)', t', (Psi(1:N,:).^2)','LineStyle','none')
title({'Stochastic 1D NLSE with additive noise', ...
       'implicit (CN) E-M Scheme'}, 'FontSize', 18);
xlabel('$x$', Interpreter='latex', FontSize=20);
ylabel('$t$', Interpreter='latex', FontSize=20);
zlabel('$|\psi|^2$', Interpreter='latex', FontSize=20);
colormap jet;
colorbarHandle = colorbar;
ylabel(colorbarHandle, '$|\psi|^2$', 'Interpreter','latex', 'FontSize',18);


%% Compute center-of-massand its statistics for this run

% squared amplitude (mass/particle density)
rho = Psi.^2;        % |psi|^2 : size N x Nt+1

% total mass over time: integral [rho(x,t)]dx = \sum_i^N [rho_i] * deltax
Mt = sum(rho, 1) * dx;

% integral [x * rho(x,t)] dx
xrho = x .* rho;
xrho_sum = sum(xrho, 1) * dx;

% centre of mass over time
x_cm = xrho_sum ./ Mt;

% display
fprintf('Mean of center of mass: %g\n', mean(x_cm));
fprintf('Variance of center of mass: %g\n', var(x_cm));

% plots
figure;
plot(t(1:size(x_cm, 2)), x_cm, 'LineWidth', 1.2);
xlabel('t', 'FontSize', 20); ylabel('x_{CM}(t)', 'FontSize', 20);
title('Centre of mass over time', 'FontSize', 20);
ax = gca;
ax.FontSize = 15;

figure;
histogram(x_cm, 40);
xlabel('x_{CM}', 'FontSize', 20); ylabel('count', 'FontSize', 20);
ax = gca; ax.FontSize = 15;
title('Histogram of centre of mass over time', 'FontSize', 20);




%% Function for defining the initial conditions for NLSE problem
function [y0] = init(x_grid, N, L)
q0 = zeros(N,1);  % imag part
p0 = zeros(N,1);  % real part 
for j = 1:N
    p0(j,1) = 0.5*(1+ (0.01*cos(2*pi*x_grid(j)/L)));
    q0(j,1) = 0;
end
y0=[p0; q0;];
end

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
    % [r, jac] = NLSE_eq_local_BC(pk1, qk1, pk, qk, dx, dt, gam, 'dirichlet', i, N);
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

%%
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



%% Consider different boudary conditions
% function [fvec, J] = NLSE_solve_BC(y_next, y_now, dx, dt, gam, N, BCtype)
function [fvec, J] = NLSE_solve_BC(y_next, y_now, dx, dt, gam, N, BCtype, pL, pR, qL, qR)
% Efficient residual and Jacobian for 1D NLSE implicit scheme with BC option
%
% Inputs:
% y_now, y_next : [2N x 1] real vectors [p; q]
% dx, dt, gam   : discretization and nonlinearity parameters
% N             : number of spatial grid points
% BCtype        : 'periodic', 'dirichlet', 'neumann'
%
% Output:
% fvec [2N x 1]  : nonlinear residual
% J    [2N x 2N] : Jacobian matrix

fvec = zeros(2*N, 1);
J = spalloc(2*N, 2*N, 12*N);

% Split p and q
p_now  = y_now(1:N);
q_now  = y_now(N+1:end);
p_next = y_next(1:N);
q_next = y_next(N+1:end);

% ------------------------------
% Extend arrays based on BC
% ------------------------------
switch lower(BCtype)
    case 'periodic'
        % Periodic extension
        p_now_ext  = [p_now(end); p_now; p_now(1)];
        q_now_ext  = [q_now(end); q_now; q_now(1)];
        p_next_ext = [p_next(end); p_next; p_next(1)];
        q_next_ext = [q_next(end); q_next; q_next(1)];
        interior_nodes = 1:N;    % interior points
    case {'dirichlet', 'neumann'}
        % No extension, skip boundaries in loop
        p_now_ext  = p_now;
        q_now_ext  = q_now;
        p_next_ext = p_next;
        q_next_ext = q_next;
        interior_nodes = 2:N-1;  % interior points
    
    otherwise
        error('Unknown BC type: %s', BCtype);
end

% ------------------------------
% Loop over interior points
% ------------------------------
for i = interior_nodes
    switch lower(BCtype)
        case 'periodic'
            idx = i:i+2;   % maps to i-1, i, i+1 in extended arrays
        case {'dirichlet','neumann'}
            idx = i-1:i+1; % standard 3-point stencil
    end

    % Collect local triplets
    pk  = p_now_ext(idx);
    qk  = q_now_ext(idx);
    pk1 = p_next_ext(idx);
    qk1 = q_next_ext(idx);

    % Local nonlinear residual and Jacobian (2x6)
    % [r, jac] = NLSE_eq_local(pk1, qk1, pk, qk, dx, dt, gam);
    [r, jac] = NLSE_eq_local_BC(pk1, qk1, pk, qk, dx, dt, gam, BCtype, i, N);

    % Fill global residual
    fvec(i)   = r(1);
    fvec(N+i) = r(2);

    % Global Jacobian indices
    if strcmpi(BCtype, 'periodic')
        i_prev = mod(i-2, N) + 1;   % i-1
        i_curr = i;                 % i
        i_next = mod(i, N) + 1;     % i+1
    else
        i_prev = i-1;
        i_curr = i;
        i_next = i+1;
    end

    i_fill = [i_prev, i_curr, i_next, ...
                N+i_prev, N+i_curr, N+i_next];
    
    % Place local Jacobian into sparse global matrix
    J(i,   i_fill) = jac(1,:);
    J(N+i, i_fill) = jac(2,:);
end


% ---------------------------------------------------
% Impose boundary conditions for Dirichlet or Neumann
% ---------------------------------------------------
switch lower(BCtype)
    case 'dirichlet'
        boundaries = [1, N, N+1, 2*N];
        
        % fvec(boundaries) = [p_next(1); p_next(N); q_next(1); q_next(N)];
        
        fvec(boundaries) = [pL; pR; qL; qR];
        J(boundaries, :) = 0;        % zero out boundary rows
        J(1,1)=1; J(N,N)=1;          % Set diagonal to 1
        J(N+1,N+1)=1; J(2*N,2*N)=1;  % Set diagonal to 1

    case 'neumann'
        % one-sided differences already inside NLSE_eq_local if desired
        % or leave fvec = 0 for residual at boundaries
        boundaries = [1, N, N+1, 2*N];
        fvec(boundaries) = 0;  
        J(boundaries, :) = 0;
        J(1,1)=1;                % Set diagonal to 1
        J(N,N)=1;
        J(N+1,N+1)=1;
        J(2*N,2*N)=1;
end

end

%%
function [residual, jac] = NLSE_eq_local_BC(pk1, qk1, pk, qk, dx, dt, gam, BCtype, i, N)
% Local 3-point Crank–Nicolson (midpoint rule) for 1D NLSE (real form)
%
% Inputs:
% pk1, qk1 : 3-point vectors at time n+1 (center + neighbors)
% pk, qk   : 3-point vectors at time n
% BCtype   : 'periodic','dirichlet','neumann'
% i        : current global index
% N        : total number of grid points
%
% Outputs:
% residual : 2x1 vector
% jac      : 2x6 matrix

% Midpoint nonlinear term (Crank–Nicolson)
pmid = 0.5 * (pk(2) + pk1(2)); % midpoint at center
qmid = 0.5 * (qk(2) + qk1(2));
r2mid = pmid^2 + qmid^2;

% ---------------------------
% Laplacian (with BC handled)
% ---------------------------
switch lower(BCtype)
    case 'periodic'
        dp_now  = (pk(1) - 2*pk(2) + pk(3)) / dx^2;
        dq_now  = (qk(1) - 2*qk(2) + qk(3)) / dx^2;
        dp_next = (pk1(1) - 2*pk1(2) + pk1(3)) / dx^2;
        dq_next = (qk1(1) - 2*qk1(2) + qk1(3)) / dx^2;

    case 'dirichlet'
        % Dirichlet: u(0)=u(L)=0, center = pk(2)
        if i==1 || i==N
            % zero Laplacian at fixed boundaries
            dp_now=0; dq_now=0; dp_next=0; dq_next=0;       
        else
            dp_now  = (pk(1) - 2*pk(2) + pk(3)) / dx^2;
            dq_now  = (qk(1) - 2*qk(2) + qk(3)) / dx^2;
            dp_next = (pk1(1) - 2*pk1(2) + pk1(3)) / dx^2;
            dq_next = (qk1(1) - 2*qk1(2) + qk1(3)) / dx^2;
        end

    case 'neumann'
        % Neumann: u'(0)=u'(L)=0, use one-sided Laplacian at boundaries
        if i==1
            % % forward difference
            % dp_now  = ( -2*pk(2) + 2*pk(3) ) / dx^2;
            % dq_now  = ( -2*qk(2) + 2*qk(3) ) / dx^2;
            % dp_next = ( -2*pk1(2) + 2*pk1(3) ) / dx^2;
            % dq_next = ( -2*qk1(2) + 2*qk1(3) ) / dx^2;

            % Second-order forward differenc
            dp_now  = (-3*pk(2)+4*pk(3)-pk(4))/ (2*dx^2);
            dq_now  = (-3*qk(2)+4*qk(3)-qk(4))/ (2*dx^2);
            dp_next = (-3*pk1(2)+4*pk1(3)-pk1(4))/ (2*dx^2);
            dq_next = (-3*qk1(2)+4*qk1(3)-qk1(4))/ (2*dx^2);
        elseif i==N
            % % backward difference
            % dp_now  = ( 2*pk(1) - 2*pk(2) ) / dx^2;    
            % dq_now  = ( 2*qk(1) - 2*qk(2) ) / dx^2;
            % dp_next = ( 2*pk1(1) - 2*pk1(2) ) / dx^2;
            % dq_next = ( 2*qk1(1) - 2*qk1(2) ) / dx^2;
            % Second-order backward difference
            dp_now  = (-3*pk(2)+4*pk(1)-pk(0))/ (2*dx^2); % use pk(0)=pk(1)?
            dq_now  = (-3*qk(2)+4*qk(1)-qk(0))/ (2*dx^2);
            dp_next = (-3*pk1(2)+4*pk1(1)-pk1(0))/ (2*dx^2);
            dq_next = (-3*qk1(2)+4*qk1(1)-qk1(0))/ (2*dx^2);
        else
            dp_now  = (pk(1) - 2*pk(2) + pk(3)) / dx^2;
            dq_now  = (qk(1) - 2*qk(2) + qk(3)) / dx^2;
            dp_next = (pk1(1) - 2*pk1(2) + pk1(3)) / dx^2;
            dq_next = (qk1(1) - 2*qk1(2) + qk1(3)) / dx^2;
        end
end

dp_avg = 0.5*(dp_now + dp_next);
dq_avg = 0.5*(dq_now + dq_next);

% ----------
% Residuals
% ----------
residual = [
    pk1(2) - pk(2) + dt * ( dq_avg + gam * r2mid * qmid );  % p-equation
    qk1(2) - qk(2) - dt * ( dp_avg + gam * r2mid * pmid )   % q-equation
];

% -------------------------
% Analytical Jacobian (2x6)
% -------------------------
deriv_Lap = [1, -2, 1] / (2*dx^2); % average of Laplacians over k, k+1
jac = zeros(2,6);

% Row 1
jac(1,4:6) = dt*deriv_Lap;
jac(1,2) = jac(1,2) + dt*gam*(2*pmid*qmid)/2 + 1;
jac(1,5) = jac(1,5) + dt*gam*(3*qmid^2 + pmid^2)/2;

% Row 2
jac(2,1:3) = -dt*deriv_Lap;
jac(2,2) = jac(2,2) - dt*gam*(3*pmid^2 + qmid^2)/2;
jac(2,5) = jac(2,5) - dt*gam*(2*pmid*qmid)/2 + 1;

end


%% Define time-varying BCs
function [pL, pR, qL, qR] = BCfunction(t, T)
    pL = 10^-3*cos(2*pi*t/T);      % left boundary for p
    pR = 10^-3*sin(2*pi*t/T);      % right boundary for p
    qL = 0;                        % left boundary for q
    qR = 0;                        % right boundary for q
end
