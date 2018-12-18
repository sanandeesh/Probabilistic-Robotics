%% ================= Special Problems Summer 2018 =========================
% Topic:    Demonstration of Simulataneous Localization & Mapping (SLAM)
%           with the Extended Kalman Filter (EKF)
% Student:  Sanandeesh Kamat (M.Sc candidate)
% Adivisor: Dr. Zoran Gajik
% School:   Rutgers University - Graduate School of Engineering
%           Electical & Computer Engineering 
% Source:   S. Thrun, "Probabilistic Robotics", MIT Press (2006)
% =========================================================================

function [] = mainEKFSLAM()
    clc;
    close all;
    bVideo = false;
    if bVideo
        hVidWrtr = VideoWriter('./EKFSLAM.avi');
        open(hVidWrtr);
    end
    Ts = 0.1;   % Time Step (s)
    v  = 2;     % Commanded Translational Velocity (m/s)
    w  = 0.1;   % Commanded Angular Velocity (rad/s)
    % Initialize True Pose
    pose = [0;  % x (m)
            0;  % y (m)
            pi]; % bearing (rad)
    trailLength = 200; 
    posTrail = zeros(2, trailLength); % Circular Buffer of Past Robot Positions
    theta = pose(3);
    % Initialize Map of Landmark Features
    numLandmarks = 10;
    rngLandmarks = 30;
    map = InitMap(rngLandmarks, numLandmarks);
    % Initialize the Estimated Posterior/Belief of State: 1st & 2nd Moments of Gaussian Rand Vector
    % First Moment of the Belief
    % First three elements are the robot pose (x_t,y_t,theta_t), the remainder are landmark coordinates (x_i, y_i)
    dimState   = 3+2*numLandmarks; % Dimensionality of SLAM State
    muState    = zeros(dimState,1); 
    muState(3) = pi;
    covState   = diag(0.1*ones(1,dimState)); % Second Moment of the Belief
    covState(1:3,1:3) = 0;
    muPosTrail = zeros(2, trailLength); % Circular Buffer of Past Robot Positions 
    muMapTrail = NaN*zeros(2*numLandmarks, trailLength); % Circular Buffer of Past Robot Positions 
    % Define Motion Error Parameters
    alphaVal = 0.1;
    alphaVec = [alphaVal, alphaVal, alphaVal, alphaVal, 0, 0];
    % Define Measurement Error Parameters
    varObs = [0.1;   % Variance of Measured Range (m^2)
              0.01]; % Variance of Measured Bearing (rad^2)
    % Initialize Figure
    time = 0;
    maxTime  = 100; % sec 
    numSteps = floor(maxTime/Ts);
    [hFig,     hCurrPos,   hCurrAz,  hTrail, ...
     hObs,     hCurrMuPos, hPostPDF, hCurrMuAz, ...
     hMuTrail, hDynText,   hEstMap, hMapTrail] = InitFigure(pose, theta, posTrail, map, v, w, alphaVec, varObs, muMapTrail);
    % ======================== Begin Simulation ======================== 
    for iStep = 1:numSteps
        if ishandle(hFig)
            time = time + Ts; % Update Time
            % =============== Motion & Measurement Models ================  
            % Generate the Next Robot Pose with the Noisy Velocity Motion Model
            [pose] = GenerateNextPose(v, w, Ts, pose, alphaVec);
            theta = pose(3);
            % Update Position Trail (Circular Buffer)
            posTrail(:, 1:end-1) =  posTrail(:,2:end);
            posTrail(:,end)      = [pose(1);   % x
                                    pose(2)];  % y
            % Generate Noisy Local Observations of Landmarks (Range/Bearing)
            obsLocal  = GenerateMeasurements(pose, map, varObs);
            obsGlobal = Local2Global(pose, obsLocal); % Transform into Global (x/y) Coordinates
            % =================== EKF SLAM ===================  
            [muState, covState] = EKFSLAM(muState, covState, v, w, obsLocal, alphaVec, varObs);
            % Compute Mapping Errors
            muMap = ones(size(map));
            muMap(1,:) = muState(4:2:end); % x coords of estimated landmarks
            muMap(2,:) = muState(5:2:end); % y coords of estimated landmarks
            mapErr = mean(mean((map(1:2,:)-muMap(1:2,:)).^2));
            muTheta = muState(3);
            % Update Est. Position Trail (Circular Buffer)
            muPosTrail(:,1:end-1) = muPosTrail(:,2:end);
            muPosTrail(:,end)     = [muState(1);   % x
                                     muState(2)];  % y 
            % Update Est. Map Trail (Circular Buffer)
            muMapTrail(:,1:end-1) = muMapTrail(:,2:end);
            muMapTrail(:,end)     = muState(4:end);
            % =================== Update Figure ===================  
            % Update True Robot Pose
            pointingVec = [pose(1), pose(1)+2*cos(theta);
                           pose(2), pose(2)+2*sin(theta)];
            set(hCurrPos, 'XData',  pose(1),          'YData', pose(2));          % Current x/y Pos
            set(hCurrAz,  'XData',  pointingVec(1,:), 'YData', pointingVec(2,:)); % Current Pointing Angle
            set(hTrail,   'XData',  posTrail(1,:),    'YData', posTrail(2,:));    % Pose Trail
            % Update Robot Measurements
            set(hObs,     'XData',  obsGlobal(1,:),   'YData', obsGlobal(2,:));
            % Update Posterior Belief
            muPointingVec = [muState(1), muState(1)+2*cos(muTheta);
                             muState(2), muState(2)+2*sin(muTheta)];
            set(hCurrMuPos, 'XData', muState(1),         'YData',  muState(2));        % Current x/y Pos Estimate
            set(hCurrMuAz,  'XData', muPointingVec(1,:),'YData',  muPointingVec(2,:)); % Current Pointing Angle Estimate
            set(hMuTrail,   'XData', muPosTrail(1,:),   'YData', muPosTrail(2,:));     % Estimated Pose Trail
            set(hEstMap,    'XData', muMap(1,:),        'YData', muMap(2,:));          % Latest Map Estimate
            for iLM = 1:numLandmarks
                set(hMapTrail(iLM), 'XData', muMapTrail(2*iLM-1,:), 'YData', muMapTrail(2*iLM,:));          % Latest Map Estimate
            end
            set(hPostPDF,   'CData', covState);                                        % Covariance Matrix of State 
            % Update Text Descriptions
            posErr = round(sqrt((muState(1)-pose(1))^2+(muState(2)-pose(2))^2), 2); % m
            angErr = round(180/pi*(muState(3)-pose(3)), 0); % deg
            [descr] = GetDynamicText(time, posErr, angErr, mapErr);
            set(hDynText, 'str', descr);
            drawnow;
            if bVideo % Write to Video
                currFrame = getframe(hFig);
                writeVideo(hVidWrtr, currFrame);
            end
        else
            break;
        end
    end
    if bVideo
        close(hVidWrtr);
    end
    return;
end

%% Transform Observations from Local (Range/Bearing) to Global (x/y) Coordinate System
function [obsGlobal] = Local2Global(pose, obsLocal)
    obsGlobal = zeros(size(obsLocal));
    numLandmarks = size(obsLocal, 2);
    for iObs = 1:numLandmarks
        %                         Range              Global Bearing
        obsGlobal(1,iObs) = obsLocal(1,iObs)*cos(obsLocal(2,iObs)+pose(3))+pose(1);
        obsGlobal(2,iObs) = obsLocal(1,iObs)*sin(obsLocal(2,iObs)+pose(3))+pose(2);
    end
    return;
end

%% Generate Measurements
function [obs] = GenerateMeasurements(pose, map, varObs)
    obs = ones(size(map));
    numLandmarks = size(map, 2);
    for iObs = 1:numLandmarks
        deltaX = map(1, iObs)-pose(1);
        deltaY = map(2, iObs)-pose(2);
        %              Deterministic                   Random Error
        obs(1, iObs) = sqrt(deltaX^2 + deltaY^2)     + normrnd(0, varObs(1)); % Measured Range (m)
        obs(2, iObs) = atan2(deltaY, deltaX)-pose(3) + normrnd(0, varObs(2)); % Measured Bearing (rad)
        % Keep Measured Bearing b/w +[0,..,2PI]                     
%         obs(2, iObs) = mod(obs(2, iObs)+2*pi, 2*pi);
    end
    return;
end

%% Initialize Map
% INPUTS
% range: range of each landmark from the origin
% numLandmarks: Number of initialized landmarks
% OUTPUTS
% map: matrix of map landmark coordiantes
function [map] = InitMap(range, numLandmarks)
    % Compute Azimuths of Landmarks (radians)
    angles = [0:(2*pi/numLandmarks):(2*pi/numLandmarks*(numLandmarks-1))];
    map = ones( 3, numLandmarks);
    % Compute x/y coordinates of each landmark (signatures are all 1)
    for iLandmark = 1:numLandmarks
        map(1,iLandmark) = range*cos(angles(iLandmark));
        map(2,iLandmark) = range*sin(angles(iLandmark));
    end
    return;
end


%% Initialize Figure
function [hFig,     hCurrPos,   hCurrAz,  hTrail, ...
          hObs,     hCurrMuPos, hPostPDF, hCurrMuAz, ...
          hMuTrail, hDynText, hEstMap, hMapTrail] = InitFigure(pose, theta, posTrail, map, v, w, alpha, varObs, mapTrail)
    s = get(0, 'ScreenSize');
    hFig = figure('Position', [0 0 s(3) s(4)]);
    % ============== Subplot 1: Robot & Landmark Positions ==============
    subplot(2, 3, [1, 2, 4, 5]);
    % True Robot Pose
    hCurrPos = scatter(pose(1), pose(2), 175, 'b', 'filled', 'MarkerEdgeColor', 'k'); hold on; 
    hCurrAz  = plot([pose(1), pose(1)+2*cos(theta)], [pose(2), pose(2)+2*sin(theta)], 'b', 'LineWidth', 3);
    hTrail   = plot( posTrail(1,:), posTrail(2,:), '--b');
    % Map Landmarks and Observations
    hMap = scatter(map(1,:), map(2,:), 125, 'r', 'filled');   
    hObs = scatter(      [],       [], 125, 'b');
    % Estimatd Robot Pose
    hCurrMuPos = scatter(pose(1), pose(2), 175, 'm', 'filled', 'pentagram', 'MarkerEdgeColor', 'k');
    hCurrMuAz  = plot([pose(1), pose(1)+2*cos(theta)], [pose(2), pose(2)+2*sin(theta)], 'm', 'LineWidth', 3);
    hMuTrail   = plot(posTrail(1,:), posTrail(2,:), '--m');
    % Axes/Text/etc.
    title('Planar Robot SLAM with Extended Kalman Filter (EKF)', 'fontsize', 20);
    xlabel('X Coordinate (m)', 'fontweight', 'bold', 'fontsize', 13);
    ylabel('Y Coordinate (m)', 'fontweight', 'bold', 'fontsize', 13);
    descr = GetDynamicText(0, 0, 0, 0);
    hDynText   = text( -60, 31, descr, 'FontSize', 13);
    descr = GetParamText(v, w, alpha, varObs);
    text( -60, 9, descr, 'FontSize', 13);
    grid on;
    axis equal tight;
    xlim([-35.5 35.5]);
    ylim([-35.5 35.5]);
    legend([hCurrPos, hCurrMuPos, hMap, hObs], 'True Pose', 'Estimated Pose', 'True Landmark', 'Measured Landmark');
    % ============== Subplot 2: Estimated Landmark Positions ==============
    subplot(2, 3, [3]);
    hEstMap = scatter(      [],       [], 125, 'b', 'filled'); hold on;
    hMapTrail = plot(mapTrail(1:2:end,:)',mapTrail(2:2:end,:)','b');
    title('Estimated Map Coordinates');
    grid on;
    axis equal tight;
    % ============== Subplot 3: Covariance Matrix ==============
    subplot(2, 3, [6]);
    hPostPDF = imagesc(zeros(100), [0.0, 0.02]);
    title('Posterior Covariance Matrix');
    colorbar;
    axis equal tight;
    colormap hot;
    return;
end

%% Generate the Next Robot Pose with the Noisy Velocity Motion Model
function [poseNext] = GenerateNextPose(v, w, Ts, poseCurr, alpha)
    % Generate Noisy Velocity Commands
    vHat   = v + normrnd(0, alpha(1)*v^2 + alpha(2)*w^2);
    wHat   = w + normrnd(0, alpha(3)*v^2 + alpha(4)*w^2);
    gamHat = 0 + normrnd(0, alpha(5)*v^2 + alpha(6)*w^2);
    % Update Pose
    theta  = poseCurr(3);
    poseNext = poseCurr + [-(vHat/wHat)*sin(theta)+(vHat/wHat)*sin(theta+wHat*Ts);
                            (vHat/wHat)*cos(theta)-(vHat/wHat)*cos(theta+wHat*Ts);
                             wHat*Ts + gamHat*Ts];
    % Keep Pose Theta in [0,..,2PI]                     
    poseNext(3) = mod(poseNext(3)+2*pi, 2*pi);                     
    return;
end

%% Get Dynamic Text: time, position error, and angle error
function [descr] = GetDynamicText(time, posErr, angErr, mapErr)
    descr = {['Time = ', num2str(time), 's'];
             ['Pos Err = ', num2str(posErr), 'm'];
             ['Ang Err = ', num2str(angErr), 'deg'];
             ['Mean Map Err = ', num2str(mapErr), 'm']};
    return;
end

%% Get Parameter Text 
function [descr] = GetParamText(v, w, alphaVec, varObs)
   descr = {  'Command Velocities';
             ['v = ', num2str(v), 'm/s'];
             ['w = ', num2str(w), 'rad/s'];
             ' '
             'Motion Error Params';
             ['\alpha_{1} = ', num2str(alphaVec(1)), ', \alpha_{2} = ', num2str(alphaVec(2))];
             ['\alpha_{3} = ', num2str(alphaVec(3)), ', \alpha_{4} = ', num2str(alphaVec(4))];
             ['\alpha_{5} = ', num2str(alphaVec(5)), ', \alpha_{6} = ', num2str(alphaVec(6))];
             ' '
             'Measurement Model';
              'Error Parameters';
              ['\sigma_{R}^{2} = ', num2str(varObs(1)), 'm^2'];
              ['\sigma_{\phi}^{2} = ', num2str(varObs(2)), 'rad^2']};             
    return;
end


