%function [dxdphi,force] = state_eq_of_phi_2(phi,x,u_ang,sct)
function [dxdphi,force] = state_eq_of_phi_3(phi,xhat,uhat,sct, Px,qx,Pu,qu)
% Nonlinear state equations
%   Calculates the derivatives with respect to phi for the state variables.
%
% phi - independent variable
% x - state vector, size [7x1], [h hx hy ex ey m t]'
% u - [alpha; beta] actuation angles;
% F - thrust. May be 0 if in eclipse.
% sct - system constatants. Includes all orbit and spacecraft parameters
% phi - [rad]

x     = Px*xhat +qx;
u_ang = Pu*uhat +qu;

% Rename the states
h = x(1); % [km^2/s]
hx = x(2); % [km^2/s]
hy = x(3); % [km^2/s]
ex = x(4); % [-]
ey = x(5); % [-]
m = x(6); % [kg]
tim = x(7); % [seconds]


% Common values used
sphi = sin(phi);
cphi = cos(phi);

A = ex*sphi - ey*cphi;
B = 1 + ex*cphi + ey*sphi;

sh2y2 = sqrt(h^2-hy^2);
hz = sqrt(h^2-hx^2-hy^2);

mum = sct.mu*m;
mumB = mum*B;


% Calculate radius of s/c in Earth centered inertial frame ECI J2000
rsc_in = h^2/(mumB/m*sh2y2)*[hz*cphi - hx*hy/h*sphi
                             (h^2-hy^2)/h*sphi
                             -hx*cphi - hy*hz/h*sphi];


% J2 perturbation
zeta = atan(hx/hz); 
seta = - hy/h; 
ceta = sh2y2/h;


% Rotations from I (ECI) -> I' -> O -> R
Rzeta = [cos(zeta) 0 -sin(zeta); 0 1 0; sin(zeta) 0 cos(zeta)]; % Rotation angle zeta
Reta = [1 0 0; 0 ceta seta; 0 -seta ceta]; % Rotation angle eta
Rphi = [cphi sphi 0; -sphi cphi 0; 0 0 1]; % Rotation angle phi
RECItoR = Rphi*Reta*Rzeta;



% Radius magnitude 
rsc = h^2/(sct.mu*B);
aJ2 = -sct.mu*sct.J2Ebar*sct.R_Earth^2*(3*rsc_in(3)/rsc^5*RECItoR(:,3) + (3/(2*rsc^4) - 15*rsc_in(3)^2/(2*rsc^6))*[1;0;0]);
FJ2 = m*aJ2; % Force due to J2 acceleration


% Moon perturbation
% rMoon = MoonSimpson_2(tim/(24*3600)); % [km] Moon position. Returns Lunar cartesian coordinates (mean equator and equinox of epoch J2000)
% rscMoon = rMoon - rsc_in; % Spacecraft to Moon position vector
% F_due_Moon = RECItoR*sct.muMoon*m*(rscMoon/norm(rscMoon)^3 - rMoon/norm(rMoon)^3); % Eq. 6 in Correct Modeling of the indirect term for third body perturbations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rMoon = MoonSimpson_2(tim); % [km] Moon position. Returns Lunar cartesian coordinates (mean equator and equinox of epoch J2000)
rscMoon = rMoon - rsc_in; % Spacecraft to Moon position vector
F_due_Moon = RECItoR*sct.muMoon*m*(rscMoon/norm(rscMoon)^3 - rMoon/norm(rMoon)^3); % Eq. 6 in Correct Modeling of the indirect term for third body perturbations
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sun perturbation
% rSun = Sun_2(tim/86400); 
% rscSun = rSun - rsc_in; % Spacecraft to Sun position vector
% F_due_Sun = RECItoR*sct.muSun*m*(rscSun/norm(rscSun)^3 - rSun/norm(rSun)^3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rSun = Sun_2(tim); 
rscSun = rSun - rsc_in; % Spacecraft to Sun position vector
F_due_Sun = RECItoR*sct.muSun*m*(rscSun/norm(rscSun)^3 - rSun/norm(rSun)^3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Force model
% % Adjust the thrust based on eclipse or not
rscEarth = - rsc_in;
aSR = asin(sct.R_Sun/norm(rscSun));
aBR = asin(sct.R_Earth/norm(rscEarth));
aD = acos(dot(rscEarth,rscSun)/(norm(rscEarth)*norm(rscSun)));
gama_L = 1/(1+exp(-sct.cs*(aD-sct.ct*(aSR+aBR))))*sct.ConsiderShadow + (1-sct.ConsiderShadow);
F = sct.F*gama_L;



% Calculate the thrust compoenents given the actuation angles
Fr =  sin(u_ang(1)).*cos(u_ang(2))*F; % Plus or minus alpha?
Fn =  cos(u_ang(1)).*cos(u_ang(2))*F;
Fh =  sin(u_ang(2))*F;

% Add perturbations to obtain total external force
Frt = Fr + FJ2(1) + F_due_Moon(1) + F_due_Sun(1);
Fnt = Fn + FJ2(2) + F_due_Moon(2) + F_due_Sun(2);
Fht = Fh + FJ2(3) + F_due_Moon(3) + F_due_Sun(3);

% Nonlinear dt/dphi
dtdphi = h^3*mumB*sh2y2/(mumB*(mumB/m)^2*sh2y2 - h^4*hy*sphi*Fht); % Nonlinear in Fh



% All derivatives of state variables
dxdphi = zeros(7,1);
dxdphi(1,1) =                                           h^2/(mumB)*Fnt*dtdphi;
dxdphi(2,1) =                                         (h*hx/(mumB)*Fnt +  (h^2*hz/(mumB*sh2y2)*sphi+h*hx*hy/(mumB*sh2y2)*cphi)*Fht)*dtdphi;
dxdphi(3,1) =                                         (h*hy/(mumB)*Fnt +                                  -h*sh2y2/(mumB)*cphi*Fht)*dtdphi;
dxdphi(4,1) = ( h*sphi/(mum)*Frt + (2*h*cphi/(mum)+h*A*sphi/(mumB))*Fnt +                             h*ey*hy*sphi/(mumB*sh2y2)*Fht)*dtdphi;
dxdphi(5,1) = (-h*cphi/(mum)*Frt + (2*h*sphi/(mum)-h*A*cphi/(mumB))*Fnt +                            -h*ex*hy*sphi/(mumB*sh2y2)*Fht)*dtdphi;
dxdphi(6,1) = - F/(sct.veff)*dtdphi; % Mass integration
% dxdphi(7,1) = dtdphi; % Time integration as well, in days. Assign last element in the vector for initialization.
dxdphi(7,1) = dtdphi/86400; % Time integration as well, in days. Assign last element in the vector for initialization.

%%%%%%%%%% new scaleup %%%%
dxdphi = Px^-1* dxdphi;
force = F;


end





