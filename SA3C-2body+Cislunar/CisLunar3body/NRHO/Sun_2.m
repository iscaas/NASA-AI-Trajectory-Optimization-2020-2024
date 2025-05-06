%--------------------------------------------------------------------------
%
% Sun: Computes the Sun's geocentric position using a low precision 
%      analytical series
%
% Input:
%   Mjd_TT    Terrestrial Time (Modified Julian Date)
% 
% Output:     
%   rSun      Solar position vector [km] with respect to the 
%             mean equator and equinox of J2000 (EME2000, ICRF)
%
% Last modified:   2015/08/12   M. Mahooti
% 
%--------------------------------------------------------------------------
function rSun = Sun_2(Mjd_TT)

MJD_J2000 = 51544.5;             % Modified Julian Date of J2000
mtokm = 1e-3; % Meters to km conversion

ep  = 1/180*pi*(84381.412/3600);     % Obliquity of J2000 ecliptic
T   = (Mjd_TT- MJD_J2000)/36525;     % Julian cent. since J2000

% Mean anomaly, ecliptic longitude and radius
M = 2*pi *  ( 0.9931267 + 99.9973583*T);                    % [rad]
L = 2*pi *  ( 0.7859444 + M/(2*pi) + (6892*sin(M)+72.0*sin(2*M)) / 1296e3); % [rad]
r = (149.619e9 - 2.499e9*cos(M) - 0.021e9*cos(2*M)) * mtokm;             % [km]

% Equatorial position vector
rSun = R_x(-ep) * [r*cos(L), r*sin(L), 0].';

end


%% Helper functions

%--------------------------------------------------------------------------
%  Input:
%    angle       angle of rotation [rad]
%
%  Output:
%    rotmat      rotation matrix
%--------------------------------------------------------------------------
function [rotmat] = R_x(angle)
C = cos(angle);
S = sin(angle);
rotmat = zeros(3,3);

rotmat(1,1) = 1.0;  rotmat(1,2) =    0.0;  rotmat(1,3) = 0.0;
rotmat(2,1) = 0.0;  rotmat(2,2) =      C;  rotmat(2,3) =   S;
rotmat(3,1) = 0.0;  rotmat(3,2) = -1.0*S;  rotmat(3,3) =   C;
end


