function [r_ECI,v_ECI] = CR3BP_3_ECI(r_CR3BP,v_CR3BP,epoch0_mjd_in_s)
% CR3BP_2_ECI Converts CR3BP coordinates to ECI, given the moon position in ECI frame (from ephemeris)
% 
% Mean moon eccentricity 0.0549

mu1 = 0.012150586632602; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Position of moon in ECI J2000
% [rM_ECI,vM_ECI] = MoonSimpson(epoch0_mjd_in_s);
%%%%%%%%%%%%%%%%%%%%%%%%%  NEW Updated Position of moon in ECI J2000
rM_ECI = MoonSimpson_2(epoch0_mjd_in_s);
dtM = 1; % [sec]
vM_ECI = (MoonSimpson_2(epoch0_mjd_in_s+dtM/86400)-MoonSimpson_2(epoch0_mjd_in_s-dtM/86400))/(2*dtM);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



DU = norm(rM_ECI);
TU = DU/norm(vM_ECI);

% Dimensionalize
r_CR3BP = r_CR3BP*DU;
v_CR3BP = v_CR3BP*DU/TU;

% Build coordinates with ii from Earth to moon, kk perpendicular to the Moon's orbital plane and jj after right hand rule.
kk = cross(rM_ECI,vM_ECI)/norm(cross(rM_ECI,vM_ECI));
ii = rM_ECI/norm(rM_ECI);
jj = cross(kk,ii);

% Rotation matrix
R_CR3BP_2_ECI = [ii,jj,kk];

% Translate and rotate
r_rot = r_CR3BP + [mu1*DU; 0; 0];
r_ECI = R_CR3BP_2_ECI*r_rot;

% Convert velocities with coriolis theorem
omega_r = norm(cross(rM_ECI,vM_ECI))/DU^2;

v0 = [0 mu1*DU*omega_r  0].';
v_ECI = R_CR3BP_2_ECI*(v0 + v_CR3BP + cross([0; 0; omega_r],r_CR3BP)); % This assumes velocity is perpendicular to omega_r and radius, i.e. trajectory is circular. According to book



end

