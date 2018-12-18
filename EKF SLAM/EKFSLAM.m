%% ================= Special Problems Summer 2018 =========================
% Topic:    Application of Bayesian Filtering to Robotic Perception Tasks
% Function: Extended Kalman Filter SLAM
% Student:  Sanandeesh Kamat (M.Sc candidate)
% Adivisor: Dr. Zoran Gajik
% School:   Rutgers University - Graduate School of Engineering
%           Electical & Computer Engineering 
% Source:   S. Thrun, "Probabilistic Robotics", MIT Press (2006)
% =========================================================================

function [muCurr, covCurr] = EKFSLAM(muPrev, covPrev, v, w, obs, alpha, varObs)
    % Initialize Variables
    Ts = 0.1; % Time Step (s)
    theta = muPrev(3); % Robot Azimuth
    % ========================= Prediction Step ========================= 
    dimState = length(covPrev);
    Fx = zeros(3, dimState);
    Fx(1:3,1:3) = eye(3);
    % Jacobian for Linearized Motion Model, Evaluated at u_t and mu_t-1
    Gt = eye(dimState) + Fx'*[0, 0, -(v/w)*cos(theta)+(v/w)*cos(theta+w*Ts); 
                              0, 0, -(v/w)*sin(theta)+(v/w)*sin(theta+w*Ts);
                              0, 0,                    0                  ]*Fx; 
    % Jacobian for Linearized Motion Model wrt Motion Parameters, Evaluated at u_t and mu_t-1
    Vt = [(-sin(theta)+sin(theta+w*Ts))/w,  (v*(sin(theta)-sin(theta+w*Ts)))/w^2+(v*cos(theta+w*Ts)*Ts)/w;
          (cos(theta)-cos(theta+w*Ts))/w,  -(v*(cos(theta)-cos(theta+w*Ts)))/w^2+(v*sin(theta+w*Ts)*Ts)/w;
                        0,                                                   Ts];  
    % Motion Noise Covariance Matrix from the Control
    Mt = [alpha(1)*v^2+alpha(2)*w^2,              0;
                       0,              alpha(3)*v^2+alpha(4)*w^2];
    % Motion Update 
    % Expectation of Predicted State
    muCurr_Pred = muPrev + Fx'*[-(v/w)*sin(theta)+(v/w)*sin(theta+w*Ts); % Pred x coord
                                 (v/w)*cos(theta)-(v/w)*cos(theta+w*Ts); % Pred y coord
                                               w*Ts];                % Pred Az ang
    muCurr_Pred(3) = mod(muCurr_Pred(3)+2*pi, 2*pi);
    % Covariance of Predicted State
    covCurr_Pred = Gt*covPrev*Gt' + Fx'*Vt*Mt*Vt'*Fx;
    % ========================= Measurement Step ========================= 
    % Covariance of Measurement
    varRng = varObs(1);
    varPhi = varObs(2);
    Qt = [varRng, 0,    0;
          0,    varPhi, 0;
          0,      0,    0.01];
    numObs = size(obs, 2);
    for iObs = 1:numObs
        % Index of this Landmark in the State Vector
        iObsState.x = 3+(2*iObs-1); 
        iObsState.y = 3+(2*iObs); 
        % If This Landmark has Not Been Observed Yet ({x,y}=0)
        if (muCurr_Pred(iObsState.x)==0 && muCurr_Pred(iObsState.y)==0)
            % Initialize Location by the projected location obtained from the corresponding range and bearing measurement (i.e. local to global).
            muCurr_Pred(iObsState.x) = muCurr_Pred(1)+ obs(1,iObs)*cos(obs(2,iObs)+muCurr_Pred(3));
            muCurr_Pred(iObsState.y) = muCurr_Pred(2)+ obs(1,iObs)*sin(obs(2,iObs)+muCurr_Pred(3));
        end   
        deltaX = muCurr_Pred(iObsState.x)-muCurr_Pred(1);
        deltaY = muCurr_Pred(iObsState.y)-muCurr_Pred(2);
        % Predicted Local Observation of Landmark
        rng = sqrt((deltaX)^2+(deltaY)^2);
        predObs = [rng;                                  % Range
                   atan2(deltaY, deltaX)-muCurr_Pred(3); % Bearing
                   1];                                   % Signature
        % The Jacobian of the Meas-Model wrt. robot pose and landmark coord  
        ht = (1/(rng^2)).*[-rng*deltaX, -rng*deltaY,    0, rng*deltaX, rng*deltaY,     0;
                               deltaY,     -deltaX, -rng,    -deltaY,     deltaX,     0;
                                    0,           0,    0,          0,          0, rng^2];
        % Transform Low Dim Jacobian to Full State Dim
        Fxj = GenerateFxj(iObs, dimState);
        Ht = ht*Fxj; % [3 x dimState]
        St = Ht*covCurr_Pred*Ht'+Qt;     % Covariance of the Observation [3x3]
        Kt = covCurr_Pred*Ht'*(St^(-1)); % Kalman Gain (Cross Covariance over Observation Covariance)
%         Kt = covCurr_Pred*Ht'/(St);
        % Linear Min Mean Square Estimator (LMMSE) of Gaussian State
        muCurr_Pred = muCurr_Pred + Kt*(obs(:,iObs)-predObs); % Expected State
        muCurr_Pred(3) = mod(muCurr_Pred(3)+2*pi, 2*pi);
        covCurr_Pred = (eye(dimState)-Kt*Ht)*covCurr_Pred;         % Covariance of State
    end
    % Set the Final 1st/2nd Moments of the Posterior Estimate
    muCurr  = muCurr_Pred;
    muCurr(3) = mod(muCurr(3)+2*pi, 2*pi);
    covCurr = covCurr_Pred;
    return;
end


%% Generate Matrix to Map Low Dim Jacobian to Full State Dim
function [Fxj] = GenerateFxj(iObs, dimState)
    iCol = 3+(2*iObs-1);
    Fxj = zeros(6, dimState); % Maps low-dim Jacobian to full state-dim
    Fxj(1:3,1:3) = eye(3);
    Fxj(4:5,iCol:(iCol+1)) = eye(2);
    return;
end

