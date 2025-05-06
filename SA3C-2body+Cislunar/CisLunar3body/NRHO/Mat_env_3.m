
classdef Mat_env_3 < handle
    
    properties (SetAccess = private)
       state = [];
       state_final = [];
       new_state = [];
    end
    

   methods
        function obj = Mat_env_2 ()      % constructor 
            obj.F=1.17/1000;%0.3115/1000; %in kilo Newtons
            obj.I_sp = 1800; % in sec
            obj.m0=2000; % kg
            obj.alpha=0.5; % in radians
            obj.beta=0.5;% in radians
        end
   end
   methods(Static)
       function result = resulting()
            global mu % defined in chkStop
            scriptPath = fileparts(fileparts(mfilename('fullpath')));
            csvFilePath = fullfile(scriptPath, 'outputs', 'csv', 'csvlist_2.dat');
            M = csvread(csvFilePath);
            state = M(1:7)';   % [h,hx,hy,ex,ey,mass,time]
            alpha = M(8);
            beta = M(9);
            F = M(10);         % N in example 1, later it converts in KN by divide by 1000 
            phi_0 = M(11);     % for GTO to moon phi_0 < phi_1,   for moon to GTO phi_1 < phi_0
            phi_1 = M(12);

            tol_inc = M(13); % tolerance of inclination +- deg
            tol_ecc = M(14); % 0.00001 here uses 0.0001 tolerance of eccentricity
            tol_a   = M(15); % tolerance of normalize a +- DU
            shadow_flag = M(16);  % [0 no-shadow, 1 shadow]
            Isp = M(17);      % [s]  in example 1500
            m0_GTO  = M(18);  % [kg] in example 1000
            state_final = M(19:25)';
            Px1 = M(26:31);
            qx1 = M(32:37);
            Pu1 = M(38:39);
            qu1 = M(40:41);
            st_time = M(42);

            sct.tol_inc = tol_inc;      
            sct.tol_ecc = tol_ecc;   
            sct.tol_a   = tol_a;    
            % Constants
            mtokm = 1e-3; % meters to kilometers Unit Conversion
            sct.R_Earth = 6378.1363; % [km] Radius of Earth 
            sct.R_Sun = 695500; % [km] % Radius of Sun
            sct.mu = 398600.44; % [km^3/s^2] % Earth's gravitational parameter
            sct.muMoon = 4902.8; % [km^3/s^2] % Moon's gravitational parameter
            sct.muSun  = 132712440041.9394; % [km^3/s^2]; DE436 % Sun's gravitational parameter
            sct.g0 =  9.80665 * mtokm; % [km/s^2]; % Gravitational acceleration on Earth
            sct.rGEO = 42164; % [km] % Radius of GEO orbit
            sct.hGEO = sqrt(sct.mu*sct.rGEO); % [km^2/s] % Specific angular momentum in GEO orbit
            sct.TGEO = 2*pi*sqrt(sct.rGEO^3/sct.mu); % Period of GEO orbit
            sct.J2Ebar = 1.082639e-3; % [-] % Earth's oblateness effect
            % Smooth Shadow Constants
            sct.cs = 289.78;
            sct.ct = 1;
            sct.ConsiderShadow = shadow_flag;
            % Spacecraft characteristics
            sct.Isp = Isp; % [s]
            sct.m0 = m0_GTO; % [kg]
            sct.F = F * mtokm ; % [kg*km/s^2] % Maximum thrust
            sct.veff = sct.Isp*sct.g0; % [km/s] Effective exhaust velocity
            % Eccentricity, semimajor axis and inclination at NRHO in the ECI frame
            sct.eccf = 0.1267 ;
            sct.af = 376118.50255;  % [km]
            sct.inclf = 18.1514;  % [deg]
            sct.xd = state_final;
            
            epochNRHO_mjd_in_day = st_time;       %epochNRHO_mjd_in_day = mjuliandate(2022,08,21,0,0,0);
            Px = diag([Px1(1),Px1(2),Px1(3),Px1(4),Px1(5),Px1(6),1]);
            qx = [qx1(1);qx1(2);qx1(3);qx1(4);qx1(5);qx1(6);epochNRHO_mjd_in_day];
            Pu = diag([Pu1(1);Pu1(2)]); 
            qu = [qu1(1);qu1(2)];

            % sct.x0scaled = Px^-1 * (sct.x0 - qx);
            % sct.xdscaled = Px^-1 * (sct.xd - qx);
            
            
            [new_state,force] = statepropg_3(state,[alpha;beta],phi_0,phi_1,sct, Px,qx,Pu,qu);
            
            
            input = [new_state(1); new_state(2); new_state(3); new_state(4); new_state(5);phi_1];
            Kep_elem = he_3_Keplerian(input, Px,qx,Pu,qu);   % [a,e,i,RAAN,argp, nurev]
            
            input_prev = [state(1); state(2); state(3); state(4); state(5);phi_0];
            Kep_elem_prev = he_3_Keplerian(input_prev, Px,qx,Pu,qu); 
%             
            result= [new_state; force; Kep_elem; Kep_elem_prev];

       end
     end

end
