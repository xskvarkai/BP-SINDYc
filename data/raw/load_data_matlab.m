clear; clc; close all

data = readtable("Floatshield.csv");
val_data = readtable("Floatshield_val.csv");
data = detrend(data);
val_data = detrend(val_data);
Ts = 0.025;

Y_id = [data.y data.omega];
U_id = data.u;

id_data = iddata([data.y(:) lowpass(data.omega(:), 0.000000001)], data.u(:), Ts, 'Name', 'id_data');

id_data.OutputName = {'y', 'omega'};
id_data.InputName  = {'u'};

val_data = iddata([val_data.y(:) lowpass(val_data.omega(:), 0.000000001)], val_data.u(:), Ts, 'Name','val_data');
val_data.OutputName = {'y', 'omega'};
val_data.InputName  = {'u'};

%%

tf1  = tfest(id_data(1:3000), 2, 1);

%%

compare(tf1, id_data(3000:end), 100)

%%
proc1 = procest(val_data, 'P3D');
%%
compare(proc1, val_data, 1000)
%%

ss1 = ssest(id_data, 2, 'DisturbanceModel', 'estimate');
%%
compare(ss1,id_data,2)

%% 1. Force the system to a standard numeric State-Space
% This strips away all the "idss" metadata that causes indexing issues
[A, B, C, D, K] = idssdata(ss1);
nx = size(A,1);

% Observer matrices
A_obs = A - K*C;
B_obs = [B - K*D, K];
C_obs = eye(nx);
D_obs = zeros(nx, size(B_obs, 2));

% Create a standard numeric SS (not an idss)
numeric_obs_sys = ss(A_obs, B_obs, C_obs, D_obs, ss1.Ts);

%% 2. Force u and y to be pure Column Vectors
u_raw = id_data.u; if iscell(u_raw), u_raw = u_raw{1}; end
y_raw = id_data.y; if iscell(y_raw), y_raw = y_raw{1}; end
t = id_data.SamplingInstants; if iscell(t), t = t{1}; end

% Ensure they are columns: size should be (N x 1) and (N x 2)
if size(u_raw, 2) > size(u_raw, 1), u_raw = u_raw'; end
if size(y_raw, 2) > size(y_raw, 1), y_raw = y_raw'; end
if size(t, 2) > size(t, 1), t = t'; end

% Combine inputs for lsim: [u, y1, y2]
u_total = [u_raw, y_raw];

%% 3. Execute lsim using the explicit output assignment
% Using [x_matrix, t_out] = lsim(...) forces MATLAB to return the 
% full N x nx trajectory matrix.
[x_matrix, t_out] = lsim(numeric_obs_sys, u_total, t);

%% 4. Final verification and Dimension fix
% Sometimes lsim returns (nx x N), we must have (N x nx) for SINDy
if size(x_matrix, 1) == nx && size(x_matrix, 2) ~= nx
    x_matrix = x_matrix';
end

fprintf('Final verification:\n');
fprintf('Time steps (N): %d\n', length(t));
fprintf('State columns: %d\n', size(x_matrix, 2));
fprintf('Matrix rows: %d\n', size(x_matrix, 1));

%% 5. Plot to be 100% sure
figure;
subplot(2,1,1);
plot(t, y_raw(:,1), 'k', t, (C(1,:)*x_matrix' + D(1,:)*u_total')', 'r--');
title('Verification of Output y');
subplot(2,1,2);
plot(t, x_matrix);
title(sprintf('The %d States (Ready for SINDy)', nx));
legend(arrayfun(@(n) sprintf('x%d',n), 1:nx, 'UniformOutput', false));
%%
figure;
subplot(2,1,1);
plot(t, y_raw(:,2), 'k', t, (C(2,:)*x_matrix' + D(2,:)*u_total')', 'r--');
title('Verification of Output y');
subplot(2,1,2);
plot(t, x_matrix);
title(sprintf('The %d States (Ready for SINDy)', nx));
legend(arrayfun(@(n) sprintf('x%d',n), 1:nx, 'UniformOutput', false));


%%% SINDY model -> x_matrix -> y1, y2