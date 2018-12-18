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

function [ParticleSet, muPose] = MonteCarloLocalization_Optimized(ParticleSet, v, w, Ts, obsLocal, map, alpha, varObs)
    % Forward Propagate the Whole Particle Set
    numParticles = size(ParticleSet, 2);
    numLM        = size(map, 2);
    ParticleSet = SampleMotionModel(ParticleSet, v, w, Ts, alpha);
    % Iterate over each landmark and compute the weights across particles.
    ParticleWeights = zeros(1, numParticles);
    for iLM = 1:numLM
        % Generate Weights for Each Particle from the Measurements
        ParticleWeights = ParticleWeights + MeasurementModel(ParticleSet, obsLocal(:,iLM), map(:,iLM), varObs);
    end
    % Normalize Weights
    ParticleWeights = NormalizeWeights(ParticleWeights);
    % Resample the Particle Set (Post-Measurement Posterior)
    [ParticleSet] = ResampleParticleSet(ParticleSet, ParticleWeights);
    % Estimate Pose as the Center of Mass
    muPose = mean(ParticleSet,2);
    return;
end


%% Normalize Weights
function [weightsNorm] = NormalizeWeights(weightsInput)
    weightsNorm = weightsInput./sum(weightsInput);
    weightsNorm((weightsNorm < 0)|isnan(weightsNorm)) = 0;
    return;
end

%% Generate the Next Robot Pose with the Noisy Velocity Motion Model
function [psOutput] = SampleMotionModel(psInput, v, w, Ts, alpha)
    zeroVec = zeros(1,size(psInput, 2));
    % Generate Noisy Velocity Commands
    vHat   = v + normrnd(zeroVec, alpha(1)*v^2 + alpha(2)*w^2);
    wHat   = w + normrnd(zeroVec, alpha(3)*v^2 + alpha(4)*w^2);
    gamHat = 0 + normrnd(zeroVec, alpha(5)*v^2 + alpha(6)*w^2);
    % Update Pose
    theta = psInput(3,:);
    psOutput = psInput + [-(vHat./wHat).*sin(theta)+(vHat./wHat).*sin(theta+wHat*Ts);
                            (vHat./wHat).*cos(theta)-(vHat./wHat).*cos(theta+wHat*Ts);
                             wHat*Ts + gamHat*Ts];
    psOutput(3,:) = mod(psOutput(3,:), 2*pi);                     
    return;
end


%% Generate Weights for Each Particle to a Single Landmark
function [Weights] = MeasurementModel(psInput, obsLocal, lm, varObs)
    % Expected Ranges/Angle from Each Particle to the Given Landmark
    deltaXs = lm(1)-psInput(1,:);
    deltaYs = lm(2)-psInput(2,:);
    ExpRanges = sqrt(deltaXs.^2 + deltaYs.^2);
    ExpAngles = atan2(deltaYs, deltaXs)-psInput(3,:);
    % Compute Weight for Each Particle to this Landmark
    Weights = pdf('Normal', obsLocal(1), ExpRanges, varObs(1)).*pdf('Normal', obsLocal(2),ExpAngles,varObs(2));
    return;
end

%% Resample the Particle Set (Post-Measurement Posterior)
function [ResampledPS] = ResampleParticleSet(ParticleSet, ParticleWeights)
    numParticles = size(ParticleSet, 2);
    indeces = 1:numParticles;
%     s = RandStream('mlfg6331_64');
    try
        iResampled = randsample(indeces, numParticles, true, ParticleWeights);
    catch ME
        rethrow(ME);
    end
    ResampledPS = ParticleSet(:,iResampled);
    return;
end
