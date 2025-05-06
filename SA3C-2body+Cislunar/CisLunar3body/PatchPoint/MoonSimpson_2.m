%**************************************************************************
%
%  A simple analytical lunar ephemeris model. Returns Lunar cartesian
%  coordinates (mean equator and equinox of epoch J2000).
%
% Input:
%  mjd        Modified Julian date (TT)
%
% Outputs:
%  r_moon    Moon position (km)
%  v_moon    Moon velocity (km/s)
%
%  Reference:
%    "An alternative lunar ephemeris model for on-board flight software use",
%    D. G. Simpson, Proceedings of the 1999 NASA/GSFC Flight Mechanics Symposium,
%    p. 175-184).
%
%  T = (JD - 2451545)/36525
%  X =  383.0e3 * SIN( 8399.685 * T + 5.381)
%      + 31.5e3 * SIN(   70.990 * T + 6.169)
%      + 10.6e3 * SIN(16728.377 * T + 1.453)
%      +  6.2e3 * SIN( 1185.622 * T + 0.481)
%      +  3.2e3 * SIN( 7143.070 * T + 5.017)
%      +  2.3e3 * SIN(15613.745 * T + 0.857)
%      +  0.8e3 * SIN( 8467.263 * T + 1.010)
%  Y =  351.0e3 * SIN( 8399.687 * T + 3.811)
%      + 28.9e3 * SIN(   70.997 * T + 4.596)
%      + 13.7e3 * SIN( 8433.466 * T + 4.766)
%      +  9.7e3 * SIN(16728.380 * T + 6.165)
%      +  5.7e3 * SIN( 1185.667 * T + 5.164)
%      +  2.9e3 * SIN( 7143.058 * T + 0.300)
%      +  2.1e3 * SIN(15613.755 * T + 5.565)
%  Z =  153.2e3 * SIN( 8399.672 * T + 3.807)
%      + 31.5e3 * SIN( 8433.464 * T + 1.629)
%      + 12.5e3 * SIN(   70.996 * T + 4.595)
%      +  4.2e3 * SIN(16728.364 * T + 6.162)
%      +  2.5e3 * SIN( 1185.645 * T + 5.167)
%      +  3.0e3 * SIN(  104.881 * T + 2.555)
%      +  1.8e3 * SIN( 8399.116 * T + 6.248)
%
%**************************************************************************
% function [r_moon,v_moon] = MoonSimpson_2(mjd)
function rMoon = MoonSimpson_2(Mjd_TT)

% century2day = 36525;
% 
% 
% xcoeffs = [383.0e3, 31.5e3, 10.6e3, 6.2e3, 3.2e3, 2.3e3, 0.8e3];
% ycoeffs = [351.0e3, 28.9e3, 13.7e3, 9.7e3, 5.7e3, 2.9e3, 2.1e3];
% zcoeffs = [153.2e3, 31.5e3, 12.5e3, 4.2e3, 2.5e3, 3.0e3, 1.8e3];
% xa = [8399.685, 70.990, 16728.377, 1185.622, 7143.070, 15613.745, 8467.263];
% xp = [5.381, 6.169, 1.453, 0.481, 5.017, 0.857, 1.010];
% ya = [8399.687, 70.997, 8433.466, 16728.380, 1185.667, 7143.058, 15613.755];
% yp = [3.811, 4.596, 4.766, 6.165, 5.164, 0.300, 5.565];
% za = [8399.672, 8433.464, 70.996, 16728.364, 1185.645, 104.881, 8399.116];
% zp = [3.807, 1.629, 4.595, 6.162, 5.167, 2.555, 6.248];
% 
% 
% t = (mjd - 51544.5)/century2day; % time in Julian centuries from J2000
% 
% xterms = xa * t + xp;
% yterms = ya * t + yp;
% zterms = za * t + zp;
% 
% r_moon(1,1) = dot(xcoeffs, sin(xterms));
% r_moon(2,1) = dot(ycoeffs, sin(yterms));
% r_moon(3,1) = dot(zcoeffs, sin(zterms));
% 
% 
% if nargout > 1 % If velocities are required
% % v_moon is just d(r_moon)/dt: [convert units to km/s]
% day2sec = 86400;
% vxcoeffs = xcoeffs.*xa;
% vycoeffs = ycoeffs.*ya;
% vzcoeffs = zcoeffs.*za;
% 
% v_moon(1,1) =  dot(vxcoeffs, cos(xterms))/(century2day * day2sec);
% v_moon(2,1) =  dot(vycoeffs, cos(yterms))/(century2day * day2sec);
% v_moon(3,1) =  dot(vzcoeffs, cos(zterms))/(century2day * day2sec);


%% new updated 
pi2 = 2*pi;          % 2pi
Rad = pi/180;        % Radians per degree
Arcs = 3600*180/pi;  % Arcseconds per radian
MJD_J2000 = 51544.5; % Modified Julian Date of J2000

% Constants
ep = 23.43929111*Rad; % Obliquity of J2000 ecliptic IAU 1976 Value
T = (Mjd_TT-MJD_J2000)/36525; % Julian cent. since J2000

% Mean elements of lunar orbit
L_0 =      ( 0.606433 + 1336.851344*T );    % Mean longitude [rev] w.r.t. J2000 equinox
l   = pi2* ( 0.374897 + 1325.552410*T );    % Moon's mean anomaly [rad]
lp  = pi2* ( 0.993133 +   99.997361*T );    % Sun's mean anomaly [rad]
D   = pi2* ( 0.827361 + 1236.853086*T );    % Diff. long. Moon-Sun [rad]
F   = pi2* ( 0.259086 + 1342.227825*T );    % Argument of latitude 


% Ecliptic longitude (w.r.t. equinox of J2000)
dL = +22640*sin(l) - 4586*sin(l-2*D) + 2370*sin(2*D) +  769*sin(2*l)...
     -668*sin(lp) - 412*sin(2*F) - 212*sin(2*l-2*D) - 206*sin(l+lp-2*D)...
     +192*sin(l+2*D) - 165*sin(lp-2*D) - 125*sin(D) - 110*sin(l+lp)...
     +148*sin(l-lp) - 55*sin(2*F-2*D);

L = pi2 * ( L_0 + dL/1296.0e3 );  % [rad]

% Ecliptic latitude
S  = F + (dL+412*sin(2*F)+541*sin(lp)) / Arcs; 
h  = F-2*D;
N  = -526*sin(h) + 44*sin(l+h) - 31*sin(-l+h) - 23*sin(lp+h)...
     +11*sin(-lp+h) - 25*sin(-2*l+F) + 21*sin(-l+F);

B = ( 18520.0*sin(S) + N ) / Arcs;   % [rad]

cosB = cos(B);

% Distance [km]
R = 385000 - 20905*cos(l) - 3699*cos(2*D-l) - 2956*cos(2*D)...
    -570*cos(2*l) + 246*cos(2*l-2*D) - 205*cos(lp-2*D)...
    -171*cos(l+2*D) - 152*cos(l+lp-2*D);

% Rotation matrix
R_x = zeros(3);
CC = cos(-ep);
SS = sin(-ep);
R_x(1,1) = 1;   
R_x(2,2) =  CC; R_x(2,3) =   SS;
R_x(3,2) = -SS; R_x(3,3) =   CC;

% Equatorial coordinates
rMoon =  R_x * [R*cos(L)*cosB, R*sin(L)*cosB, R*sin(B)].';

end

