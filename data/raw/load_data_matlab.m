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
<<<<<<< Updated upstream
ss1 = ssest(id_data, 4);
%%
compare(ss1,val_data,20)
=======
arx1 = nlarx(id_data, [2 1 10]);
>>>>>>> Stashed changes
