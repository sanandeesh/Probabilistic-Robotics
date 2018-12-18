%% ================= Special Problems Summer 2018 =========================
% Topic:    Application of Bayesian Filtering to Robotic Perception Tasks
% Function: Extended Kalman Filter Localization
% Student:  Sanandeesh Kamat (M.Sc candidate)
% Adivisor: Dr. Zoran Gajik
% School:   Rutgers University - Graduate School of Engineering
%           Electical & Computer Engineering 
% Source:   S. Thrun, "Probabilistic Robotics", MIT Press (2006)
% =========================================================================

function [muCurr, covCurr, pObs] = EKFLocalization(muPrev, covPrev, v, w, obs, map, alpha, varObs)
    % Initialize Variables
    Ts = 0.1; % Time Step (s)
    theta = muPrev(3); % Robot Azimuth
    % ========================= Prediction Step ========================= 
    % Jacobian for Linearized Motion Model, Evaluated at u_t and mu_t-1
    Gt = [1, 0, -(v/w)*cos(theta)+(v/w)*cos(theta+w*Ts); 
          0, 1, -(v/w)*sin(theta)+(v/w)*sin(theta+w*Ts);
          0, 0,                    1                  ]; 
    % Jacobian for Linearized Motion Model wrt Motion Parameters, Evaluated at u_t and mu_t-1
    Vt = [(-sin(theta)+sin(theta+w*Ts))/w,  (v*(sin(theta)-sin(theta+w*Ts)))/w^2+(v*cos(theta+w*Ts)*Ts)/w;
          (cos(theta)-cos(theta+w*Ts))/w,  -(v*(cos(theta)-cos(theta+w*Ts)))/w^2+(v*sin(theta+w*Ts)*Ts)/w;
                        0,                                                   Ts];  
    % Motion Noise Covariance Matrix from the Control
    Mt = [alpha(1)*v^2+alpha(2)*w^2,              0;
                       0,              alpha(3)*v^2+alpha(4)*w^2];
    % Motion Update 
    % Expectation of Predicted State
    muCurr_Pred = muPrev + [-(v/w)*sin(theta)+(v/w)*sin(theta+w*Ts); % Pred x coord
                             (v/w)*cos(theta)-(v/w)*cos(theta+w*Ts); % Pred y coord
                                               w*Ts];                % Pred Az ang
    % Covariance of Predicted State
    covCurr_Pred = Gt*covPrev*Gt' + Vt*Mt*Vt';
    % ========================= Measurement Step ========================= 
    % Covariance of Measurement
    varRng = varObs(1);
    varPhi = varObs(2);
    Qt = [varRng, 0,    0;
          0,    varPhi, 0;
          0,      0,    1];
    numObs = size(obs, 2);
    for iObs = 1:numObs
        deltaX = map(1,iObs)-muCurr_Pred(1);
        deltaY = map(2,iObs)-muCurr_Pred(2);
        % Predicted Local Observation of Landmark
        rng = sqrt((deltaX)^2+(deltaY)^2);
        predObs = [rng;                                  % Range
                   atan2(deltaY, deltaX)-muCurr_Pred(3); % Bearing
                   1];                                   % Signature
        Ht = [-(deltaX)/rng,     -(deltaY)/rng,      0;
               (deltaY)/(rng^2), -(deltaX)/(rng^2), -1;
                       0,                 0,         0];
        St = Ht*covCurr_Pred*Ht'+Qt;     % Covariance of the Observation
        Kt = covCurr_Pred*Ht'*(St^(-1)); % Kalman Gain (Cross Covariance over Observation Covariance)
        % Linear Min Mean Square Estimator (LMMSE) of Gaussian State
        muCurr_Pred = muCurr_Pred + Kt*(obs(:,iObs)-predObs); % Expected State
        covCurr_Pred = (eye(3)-Kt*Ht)*covCurr_Pred;         % Covariance of State
    end
    % Set the Final 1st/2nd Moments of the Posterior Estimate
    muCurr  = muCurr_Pred;
    covCurr = covCurr_Pred;
    % Compute the Likelihood of the Observation Vector
    pObs  = 1;
    return;
end

