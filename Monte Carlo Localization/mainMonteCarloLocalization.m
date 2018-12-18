%% ================= Special Problems Fall 2018 =========================
% Topic:    Monte Carlo Localization 
%           Particle filter applied to the planar robot localization
%           problem.
% Student:  Sanandeesh Kamat (M.Sc candidate)
% Adivisor: Dr. Zoran Gajik
% School:   Rutgers University - Graduate School of Engineering
%           Electical & Computer Engineering 
% Source:   S. Thrun, "Probabilistic Robotics", MIT Press (2006)
% =========================================================================

function [] = mainMonteCarloLocalization()
    clc;
    close all;
    bVideo = false;
    if bVideo
        hVidWrtr = VideoWriter('./MonteCarloLocalization.avi');
        open(hVidWrtr);
    end
    Ts = 0.10;   % Time Step (s)
    v  = 3;     % Commanded Translational Velocity (m/s)
    w  = 0.2;   % Commanded Angular Velocity (rad/s)
    % Initialize True Pose
    pose = [0;  % x (m)
            0;  % y (m)
            0]; % bearing (rad)
    trailLength = 200; 
    posTrail = zeros(2, trailLength); % Circular Buffer of Past Robot Positions
    theta = pose(3);
    % Initialize the Estimated Posterior: A Collection of Particles 
    % (i.e. unique realizations, randomly generated samples)
    numParticles = 500;
    % Particle Sample:
    % 1. X coord
    % 2. Y coord
    % 3. Theta
%     ParticleSet = zeros(3, numParticles);
    ParticleSet = random('Normal',0,1.0,3,numParticles);
    muPosTrail = zeros(2, trailLength); % Circular Buffer of Past Robot Positions
    % Define Motion Error Parameters
    alphaVal = 0.3;
    alphaVec = [alphaVal, alphaVal, alphaVal, alphaVal, 0, 0];
    % Initialize Map of Landmark Features
    map = InitMap();
    varObs = [0.9;   % Variance of Measured Range
              0.15]; % Variance of Measured Bearing
    % Initialize Figure
    time = 0;
    maxTime  = 30; % sec 
    numSteps = floor(maxTime/Ts);
    [hFig,     hCurrPos,   hCurrAz,  hTrail, ...
     hObs,     hCurrMuPos, hCurrMuAz, ...
     hMuTrail, hDynText, hKDE, hParticles] = InitFigure(pose, theta, posTrail, map, v, w, alphaVec, varObs);
    % ======================== Begin Simulation ======================== 
    for iStep = 1:numSteps
        if ishandle(hFig)
            time = time + Ts; % Update Time
            % =============== Motion & Measurement Models ================  
            % Generate the Next Robot Pose with the Noisy Velocity Motion Model
            [pose] = GenerateNextPose(v, w, Ts, pose, alphaVec);
            % Update Position Trail (Circular Buffer)
            posTrail(:, 1:end-1) =  posTrail(:,2:end);
            posTrail(:,end)      = [pose(1);   % x
                                    pose(2)];  % y
            % Generate Noisy Local Observations of Landmarks (Range/Bearing)
            obsLocal  = GenerateMeasurements(pose, map, varObs);
            obsGlobal = Local2Global(pose, obsLocal); % Transform into Global (x/y) Coordinates
            % ================= Monte Carlo Localization =================
            [ParticleSet, muPose] = MonteCarloLocalization(ParticleSet, v, w, Ts, obsLocal, map, alphaVec, varObs);
            % Update Position Trail (Circular Buffer)
            muPosTrail(:,1:end-1) = muPosTrail(:,2:end);
            muPosTrail(:,end)     = [muPose(1);   % x
                                     muPose(2)];  % y
            [xPost, yPost, posteriorKDE] = ComputeKDEParticleSet(ParticleSet, muPose);                       
            % ============================================================
            % =================== Update Figure ===================  
            % Subplot: Full Area
            % Update True Robot Pose
            pointingVec = [pose(1), pose(1)+2*cos(pose(3));
                           pose(2), pose(2)+2*sin(pose(3))];
            set(hCurrPos, 'XData',  pose(1),          'YData', pose(2));          % Current x/y Position
            set(hCurrAz,  'XData',  pointingVec(1,:), 'YData', pointingVec(2,:)); % Current Pointing Angle
            set(hTrail,   'XData',  posTrail(1,:),    'YData', posTrail(2,:));    % Current Pointing Angle
            % Update Robot Measurements
            set(hObs,     'XData',  obsGlobal(1,:),   'YData', obsGlobal(2,:));
            % Update Posterior Belief
            muPointingVec = [muPose(1), muPose(1)+2*cos(muPose(3));
                             muPose(2), muPose(2)+2*sin(muPose(3))];
            set(hCurrMuPos, 'XData', muPose(1),         'YData',  muPose(2));
            set(hCurrMuAz,  'XData', muPointingVec(1,:),'YData',  muPointingVec(2,:));
            set(hMuTrail,   'XData', muPosTrail(1,:),   'YData',  muPosTrail(2,:));    % Current Pointing Angle
            set(hKDE,       'XData', xPost,             'YData', yPost, 'CData', posteriorKDE);
%             set(hParticles, 'XData', ParticleSet(1,:),  'YData', ParticleSet(2,:));   
            % Update Text Descriptions
            posErr = round(sqrt((muPose(1)-pose(1))^2+(muPose(2)-pose(2))^2), 2); % m
            angErr = round(180/pi*(muPose(3)-pose(3)), 0); % deg
            [descr] = GetDynamicText(time, posErr, angErr);
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

%% Compute an Image of the Posterior KDE
function [xPost, yPost, posteriorKDE] = ComputeKDEParticleSet(ParticleSet, muPose)
    % Determine X/Y Range of KDE
    buff = 1.0;
%     xPost = linspace(min(PreReSample(1,:)-buff), max(PreReSample(1,:)+buff), 100);
%     yPost = linspace(min(PreReSample(2,:)-buff), max(PreReSample(2,:)+buff), 100);
    xPost = linspace(muPose(1,:)-buff, muPose(1,:)+buff, 100);
    yPost = linspace(muPose(2,:)-buff, muPose(2,:)+buff, 100);
    [xGrid, yGrid] = meshgrid(xPost, yPost);
    % Generate a 2-D Gaussian for each Particle
    varParticle = 0.01; % m^2
    covParticle = [varParticle,     0;
                   0,           varParticle];
    posteriorKDE = zeros(size(xGrid(:)));
    numParticles = size(ParticleSet, 2);
    for iPrtcl = 1:numParticles
        thisPDF = mvnpdf([xGrid(:) yGrid(:)],ParticleSet(1:2,iPrtcl)',covParticle);
        posteriorKDE = posteriorKDE + thisPDF./numParticles;
    end
    posteriorKDE = reshape(posteriorKDE,length(xPost),length(xPost));
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
    end
    return;
end

%% Initialize Map
function [map] = InitMap()
    map = [-20,   0,  20, -20, 20, -20,  0, 20;  % x 
           -10, -10, -10,  10, 10,  30, 30, 30;  % y 
             1,   1,   1,   1,  1,   1,  1,  1]; % signature (unused) 
    return;
end


%% Initialize Figure
% hDynText
function [hFig,     hCurrPos,   hCurrAz,  hTrail, ...
          hObs,     hCurrMuPos, hCurrMuAz, ...
          hMuTrail, hDynText,   hKDE, hParticles] = InitFigure(pose, theta, posTrail, map, v, w, alpha, varObs)
    s = get(0, 'ScreenSize');
    hFig = figure('Position', [0 0 s(3) s(4)]);
    subplot(2, 3, [1, 2, 4, 5]);
    % True Robot Pose
    hCurrPos = scatter(pose(1), pose(2), 175, 'b', 'filled', 'MarkerEdgeColor', 'k'); hold on; 
    hCurrAz  = plot([pose(1), pose(1)+2*cos(theta)], [pose(2), pose(2)+2*sin(theta)], 'b', 'LineWidth', 3);
    hTrail   = plot( posTrail(1,:), posTrail(2,:), '--b');
    % Map Landmarks and Observations
    hMap = scatter(map(1,:), map(2,:), 125, 'r', 'filled');   
    hObs = scatter(      [],       [], 125, 'b');
    % Estimated Robot Pose
    hCurrMuPos = scatter(pose(1), pose(2), 175, 'm', 'filled', 'pentagram', 'MarkerEdgeColor', 'k');
    hCurrMuAz  = plot([pose(1), pose(1)+2*cos(theta)], [pose(2), pose(2)+2*sin(theta)], 'm', 'LineWidth', 3);
    hMuTrail   = plot(posTrail(1,:), posTrail(2,:), '--m');
%     hParticles = scatter([], [], 25, 'k', 'filled');
    % Axes/Text/etc.
    title('Planar Robot Localization with a Particle Filter', 'fontsize', 15);
    xlabel('X Coordinate (m)', 'fontweight', 'bold', 'fontsize', 13);
    ylabel('Y Coordinate (m)', 'fontweight', 'bold', 'fontsize', 13);
    descr = GetDynamicText(0, 0, 0);
    hDynText   = text( -60, 41, descr, 'FontSize', 13);
    descr = GetParamText(v, w, alpha, varObs);
    text( -60, 19, descr, 'FontSize', 13);
%     [MotionModDescr, VelCommDescr, MeasurementDescr, EKFDescr] = GetEqnText();
%     descrCol = 40;
%     text(  descrCol, 48, 'Velocity Motion Model', 'FontSize', 12, 'FontWeight', 'bold');
%     text(  descrCol, 44, strcat(MotionModDescr(1), MotionModDescr(2), MotionModDescr(3)), 'FontSize', 10,'Interpreter','latex');
%     text(  75, 44, strcat(VelCommDescr(1), VelCommDescr(2), VelCommDescr(3)), 'FontSize', 10,'Interpreter','latex');
%     text(  descrCol, 38, 'Feature Measurement Model', 'FontSize', 12, 'FontWeight', 'bold');
%     text(  descrCol, 34, strcat(MeasurementDescr(1), MeasurementDescr(2), MeasurementDescr(3)), 'FontSize', 10,'Interpreter','latex');
%     text(  descrCol, 28, 'EKF Algorithm', 'FontSize', 12, 'FontWeight', 'bold');
%     text(  descrCol, 22, EKFDescr, 'FontSize', 11,'Interpreter','latex');
    grid on;
    axis equal tight;
    xlim([-35.5 35.5]);
    ylim([-25.5 45.5]);
    legend([hCurrPos, hCurrMuPos, hMap, hObs], 'True Pose', 'Estimated Pose', 'True Landmark', 'Measured Landmark');
    % KDE of Posterior Particle Set
    axKDE = subplot(2, 3, [3 6]);
    hKDE       = imagesc(); hold on;
    hParticles = scatter([], [], 15, 'c', 'filled');
    hPreSmpl   = scatter([], [], 15, 'c');
    title({'Gaussian Kernel Density Estimate','of the Posterior Particle Set'});
    xlabel('X Coordinate (m)');
    ylabel('Y Coordinate (m)');
    set(gca,'YDir','normal');
    axis equal tight;
    grid on;
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
    poseNext(3) = mod(poseNext(3), 2*pi);                     
    return;
end

%% Get Dynamic Text: time, position error, and angle error
function [descr] = GetDynamicText(time, posErr, angErr)
    descr = {['Time = ', num2str(time), 's'];
             ['Pos Err = ', num2str(posErr), 'm'];
             ['Ang Err = ', num2str(angErr), 'deg']};
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

%% Get Equation Text
function [MotionModDescr, VelCommDescr, MeasurementDescr, EKFDescr] = GetEqnText()
    % Description of Motion Model
    MotionModDescr = {'$$ \left( \begin{array}{c} x_{t} \\ y_{t} \\ \theta_{t} \end{array}\right) $$',...
                  '$$ = \left(\begin{array}{c} x_{t-1} \\ y_{t-1} \\ \theta_{t-1} \end{array}\right) $$',...
                  '$$ + \left(\begin{array}{c} -\frac{\hat{v}}{\hat{w}}\sin(\theta) + \frac{\hat{v}}{\hat{w}}\sin(\theta + \hat{w}\Delta t) \\ \frac{\hat{v}}{\hat{w}}\cos(\theta) - \frac{\hat{v}}{\hat{w}}\cos(\theta + \hat{w}\Delta t) \\ \hat{w}\Delta t + \hat{\gamma}\Delta t \end{array}\right) $$'};
    VelCommDescr = {'$$ \left(\begin{array}{c} \hat{v} \\ \hat{w} \\ \hat{\gamma} \end{array}\right) $$',...
                    '$$ = \left(\begin{array}{c} v \\ w \\ 0 \end{array}\right) $$',...
                    '$$ + \left(\begin{array}{c} \epsilon_{\alpha_{1}v^{2}+\alpha_{2}w^{2}} \\ \epsilon_{\alpha_{3}v^{2}+\alpha_{4}w^{2}} \\ \epsilon_{\alpha_{5}v^{2}+\alpha_{6}w^{2}} \end{array}\right)$$'};

    MeasurementDescr = {'$$ \left(\begin{array}{c} r_{r}^{i} \\ \phi_{t}^{i} \\ s_{t}^{i} \end{array}\right)$$',...
                        '$$ = \left(\begin{array}{c} \sqrt{(m_{j,x}-x_t)^2+(m_{j,y}-y_t)^2}   \\ atan2(m_{j,y}-y_t, m_{j,x}-x_t) - \theta \\ s_{j} \end{array}\right)$$',...
                        '$$ + \left(\begin{array}{c} \epsilon_{\sigma_{r}^{2}} \\ \epsilon_{\sigma_{\phi}^{2}} \\ \epsilon_{\sigma_{s}^{2}} \end{array}\right)$$'};
    
    EKFDescr = {'$$ \overline{\mu}_{t} = g(u_{t}, \mu_{t-1}) $$ Prediction Update (use motion model)',...
            '$$ \overline{\Sigma}_{t} = G_{t}\Sigma_{t-1}G^{T}_{t}+R_{t} $$ Covariance of Prediction (use Jacobians of motion model)',...
            '$$ K_{t} = \overline{\Sigma}_{t}H^{t}_{T}(H_{t}\overline{\Sigma}_{t}H^{T}_{t}+Q_{t})^{-1} $$ Kalman Gain (use Jacobians of measurement model)',...
            '$$ \mu_{t} = \overline{\mu}_{t} + K_{t}(z_t-h(\overline{\mu}_{t})) $$ Measurement Update (use measurement model)',...
            '$$ \Sigma_{t} = (I-K_{t}H_{t})\overline{\Sigma}_{t} $$ Covariance of the Corrected Estimate'};
    return;
end


