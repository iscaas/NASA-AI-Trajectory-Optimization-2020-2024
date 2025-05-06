function Kep_elem = he_3_Keplerian(he_vec1, Pxx,qx,Pu,qu)
% HE_2_KEPLERIAN Converts h-e elements to Keplerian elements (Earth centered). 
% Assumes prograde motion (hz positive) and mu = 398600.44 km^3/s^2 as the gravitational parameter for Earth.
% 
% Inputs:
% he_vec = a 6x1 vector: [h, hx, hy, ex, ey, phi].' in km^2/s and radians
%
% Outputs:
% Kep_elem = a 6x1 vector: [a ecc incl RAAN argp nu].' in km for semi-major axis and degrees for angles
%
% Warning! The function is not vectorized.
% Note: the built-in matlab function ijk2keplerian might internally use a different gravitaional parameter.
%

Px = Pxx(1:5,1:5);
he_vec = Px*he_vec1(1:5) +qx(1:5);


muE = 398600.44; % Gravitational parameter of Earth, km^3/s^2. Matlab uses 398600.4418 internally so inconsistencies might occur if converting back and forth with matlab built-in functions.

h = he_vec(1);
hx = he_vec(2);
hy = he_vec(3);
ex = he_vec(4);
ey = he_vec(5);
phi = he_vec1(6);

sphi = sin(phi);
cphi = cos(phi);
A = ex*sphi - ey*cphi;
B = 1 + ex*cphi + ey*sphi;
sh2y2 = sqrt(h^2-hy^2);

ecc_sq = ex^2+ey^2;
ecc = sqrt(ecc_sq);
p = h^2/muE;
a = p/(1-ecc_sq);
 
hz = sqrt(h^2-hx^2-hy^2);
incl = acosd(hz/h) ; 

vr = muE*A/h;
% vn = muE*B/h;
r_ECI = h^2/(muE*B*sh2y2) * [hz*cphi - hx*hy/h*sphi
                             (h^2-hy^2)/h*sphi
                             -hx*cphi - hy*hz/h*sphi];

% Rotations from I (ECI) -> I' -> O
zeta = atan(hx/hz);
seta = - hy/h;
ceta = sqrt(h^2-hy^2)/h;
Rzeta = [cos(zeta) 0 -sin(zeta); 0 1 0; sin(zeta) 0 cos(zeta)]; % Rotation angle zeta
Reta = [1 0 0; 0 ceta seta; 0 -seta ceta]; % Rotation angle eta
RECItoR = Reta*Rzeta;

e_ECI = RECItoR.'*[ex; ey; 0];
n = cross([0;0;1], [hx; hy; hz]);

if n(2) >= 0
    RAAN = acosd(n(1)/norm(n));
else
    RAAN = 360 - acosd(n(1)/norm(n));
end

if e_ECI(3) >= 0
    argp = acosd(dot(n,e_ECI)/(norm(n)*norm(e_ECI)));
else
    argp = 360 - acosd(dot(n,e_ECI)/(norm(n)*norm(e_ECI)));
end

if vr >= 0
    nu = acosd(dot(r_ECI,e_ECI)/(norm(r_ECI)*norm(e_ECI)));
else
    nu = 360 - acosd(dot(r_ECI,e_ECI)/(norm(r_ECI)*norm(e_ECI)));
end


Kep_elem = [a ecc incl RAAN argp nu].';

end

