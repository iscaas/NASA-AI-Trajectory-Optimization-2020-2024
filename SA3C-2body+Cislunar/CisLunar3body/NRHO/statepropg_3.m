function [x1,force] = statepropg_3(x0,u0,phi0,phi1,sct, Px,qx,Pu,qu)
% State propagation (1 step)
% Given the states x0 and controls u0 at phi0,
% find the new states x1 at phi1 by a trapezoidal explicit integration (second order Runge-Kutta scheme)
%
% Assumption: phi1 has to be "close" to phi0
% u0 = column vector [2 x 1]
% x0 = [7 x 1]
%

h = phi1 - phi0; % Step size of integration 
[k1,force] = state_eq_of_phi_3(phi0 ,x0 , u0, sct, Px,qx,Pu,qu);
[k2,force] = state_eq_of_phi_3(phi1, x0 + k1*h, u0, sct, Px,qx,Pu,qu);
x1 = x0 + h/2*(k1 + k2);


end