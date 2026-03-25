%% Urban Village Fire Risk Prediction
% GWO-LSTM with Incremental Learning
% -------------------------------------------------------------------------
% This script demonstrates the full workflow used in the study:
% 1) Load matrix-formatted time-series samples from Excel
% 2) Standardize the input data
% 3) Split data into training / validation / testing sets
% 4) Use Grey Wolf Optimizer (GWO) to search LSTM hyperparameters
% 5) Train the final LSTM model
% 6) Simulate incremental learning on streaming test batches
% 7) Export prediction results
%
% Expected Excel format:
% - Sheet 1 to Sheet 100: each sheet is a 3 x 30 matrix
% - Sheet "Y": target values (100 x 1 or 1 x 100)
%
% Replace the sample file with your own dataset following the same structure.
% -------------------------------------------------------------------------

clear; clc; close all;
rng(42);

%% 1. Load data from Excel
filename = fullfile('sample_data', 'matrices_example.xlsx');
sheets = sheetnames(filename);

numSamples = 100;
numFeatures = 3;
numTimeSteps = 30;

X = zeros(numFeatures, numTimeSteps, numSamples);
Y = zeros(numSamples, 1);

for i = 1:numSamples
    data = xlsread(filename, sheets{i});
    if size(data, 1) == numFeatures && size(data, 2) == numTimeSteps
        X(:, :, i) = data;
    else
        error('Invalid data size in sheet %s: expected %dx%d, got %dx%d.', ...
            sheets{i}, numFeatures, numTimeSteps, size(data,1), size(data,2));
    end
end

Y = xlsread(filename, 'Y');
Y = Y(:);

%% 2. Standardize data
X_mean = mean(X, 3);
X_std = std(X, 0, 3);
X_std(X_std == 0) = 1;
X = (X - X_mean) ./ X_std;

if any(isnan(X(:))) || any(isinf(X(:)))
    error('Data contains NaN or Inf values after standardization.');
end

%% 3. Split data: 70%% train, 15%% validation, 15%% test
numTrainSamples = round(0.7 * numSamples);
numValidationSamples = round(0.15 * numSamples);
numTestSamples = numSamples - numTrainSamples - numValidationSamples;

indices = randperm(numSamples);

XTrain = X(:, :, indices(1:numTrainSamples));
YTrain = Y(indices(1:numTrainSamples));

XValidation = X(:, :, indices(numTrainSamples+1:numTrainSamples+numValidationSamples));
YValidation = Y(indices(numTrainSamples+1:numTrainSamples+numValidationSamples));

XTest = X(:, :, indices(numTrainSamples+numValidationSamples+1:end));
YTest = Y(indices(numTrainSamples+numValidationSamples+1:end));

XTrainCell = convertToCellArray(XTrain);
XValidationCell = convertToCellArray(XValidation);
XTestCell = convertToCellArray(XTest);

%% 4. GWO hyperparameter optimization
lb = [10, 0.0001, 10];
ub = [200, 0.01, 200];
dim = 3;
SearchAgents_no = 10;
Max_iter = 1;
mutationProbability = 0.1; %#ok<NASGU>
mutationFactor = 0.1; %#ok<NASGU>

[best_params, best_score] = gwo(@objective_function, lb, ub, dim, SearchAgents_no, Max_iter, ...
    XTrainCell, YTrain, XValidationCell, YValidation);

fprintf('Best parameters: numHiddenUnits=%d, initialLearnRate=%.4f, maxEpochs=%d\n', ...
    round(best_params(1)), best_params(2), round(best_params(3)));
fprintf('Best validation RMSE: %.4f\n', best_score);

%% 5. Train final model with best parameters and evaluate on test set
[~, YPred] = objective_function(best_params, XTrainCell, YTrain, XTestCell, YTest);

mae = mean(abs(YPred - YTest));
r2 = 1 - sum((YPred - YTest).^2) / sum((YTest - mean(YTest)).^2);
mse = mean((YPred - YTest).^2);

fprintf('Test MAE: %.4f, R^2: %.4f, MSE: %.4f\n', mae, r2, mse);

predError = YPred - YTest;
writematrix(YPred, fullfile('results_example', 'GWO_LSTM_predictions.xlsx'));
writematrix(predError, fullfile('results_example', 'GWO_LSTM_errors.xlsx'));

figure;
plot(YPred, 'r-v', 'LineWidth', 1, 'MarkerFaceColor', 'r');
hold on;
plot(YTest, 'm-o', 'LineWidth', 1, 'MarkerFaceColor', 'm');
legend('Predicted Values', 'Actual Values');
xlabel('Test Sample Number');
ylabel('Indicator Value');
title('GWO-LSTM Prediction Results');

%% 6. Incremental learning
fprintf('\n========== Incremental Learning Starts ==========\n');

numHidden = round(best_params(1));
learnRate = best_params(2);
maxEpochs = round(best_params(3));

baseLayers = [
    sequenceInputLayer(size(XTrainCell{1},1))
    lstmLayer(numHidden,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];

baseOptions = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', learnRate, ...
    'MiniBatchSize', 32, ...
    'Verbose', false);

fprintf('Training base model...\n');
baseNet = trainNetwork(XTrainCell, YTrain, baseLayers, baseOptions);

incLearningRate = 1e-4;
incEpochs = 8;
incBatchSize = 10;
replayRatio = 0.2;
driftThreshold = 1.10;

numReplaySamples = round(0.2 * length(YTrain));
replayIndices = randperm(length(YTrain), numReplaySamples);
historicalX = XTrainCell(replayIndices);
historicalY = YTrain(replayIndices);

currentNet = baseNet;
predictions_inc = [];
update_count = 0;
rmse_history = [];

batchSize = 5;
for i = 1:batchSize:length(YTest)
    endIdx = min(i + batchSize - 1, length(YTest));
    batchX = XTestCell(i:endIdx);
    batchY = YTest(i:endIdx);

    pred = predict(currentNet, batchX);
    predictions_inc = [predictions_inc; pred]; %#ok<AGROW>

    currentRMSE = sqrt(mean((pred - batchY).^2));
    rmse_history = [rmse_history; currentRMSE]; %#ok<AGROW>

    if length(rmse_history) >= 5
        avgRMSE = mean(rmse_history(1:end-1));
        if currentRMSE > avgRMSE * driftThreshold
            fprintf('\n[Update #%d] Drift detected. Current RMSE: %.4f > threshold: %.4f\n', ...
                update_count + 1, currentRMSE, avgRMSE * driftThreshold);

            numReplay = round(replayRatio * length(batchY));
            numReplay = min(numReplay, length(historicalY));

            if numReplay > 0
                replayIdx = randperm(length(historicalY), numReplay);
                updateX = [batchX; historicalX(replayIdx)];
                updateY = [batchY; historicalY(replayIdx)];
            else
                updateX = batchX;
                updateY = batchY;
            end

            incOptions = trainingOptions('adam', ...
                'MaxEpochs', incEpochs, ...
                'InitialLearnRate', incLearningRate, ...
                'MiniBatchSize', incBatchSize, ...
                'Verbose', false);

            currentNet = trainNetwork(updateX, updateY, currentNet.Layers, incOptions);
            update_count = update_count + 1;

            newPred = predict(currentNet, batchX);
            predictions_inc(end-length(batchY)+1:end) = newPred;
            newRMSE = sqrt(mean((newPred - batchY).^2));
            fprintf('Updated RMSE after incremental training: %.4f\n', newRMSE);

            keepNum = round(0.2 * length(batchY));
            if keepNum > 0
                keepIdx = randperm(length(batchY), keepNum);
                historicalX = [historicalX; batchX(keepIdx)];
                historicalY = [historicalY; batchY(keepIdx)];
            end

            if length(historicalY) > 500
                keepIdx = randperm(length(historicalY), 500);
                historicalX = historicalX(keepIdx);
                historicalY = historicalY(keepIdx);
            end
        end
    end
end

mae_inc = mean(abs(predictions_inc - YTest(1:length(predictions_inc))));
mse_inc = mean((predictions_inc - YTest(1:length(predictions_inc))).^2);
r2_inc = 1 - sum((predictions_inc - YTest(1:length(predictions_inc))).^2) / ...
         sum((YTest(1:length(predictions_inc)) - mean(YTest(1:length(predictions_inc)))).^2);

fprintf('\n========== Incremental Learning Results ==========\n');
fprintf('Number of model updates: %d\n', update_count);
fprintf('MAE: %.4f\n', mae_inc);
fprintf('MSE: %.4f\n', mse_inc);
fprintf('R^2: %.4f\n', r2_inc);

writematrix(predictions_inc, fullfile('results_example', 'GWO_LSTM_IL_predictions.xlsx'));
writematrix(predictions_inc - YTest(1:length(predictions_inc)), ...
    fullfile('results_example', 'GWO_LSTM_IL_errors.xlsx'));

figure;
plot(predictions_inc, 'g-v', 'LineWidth', 1, 'MarkerFaceColor', 'g');
hold on;
plot(YTest(1:length(predictions_inc)), 'b-o', 'LineWidth', 1, 'MarkerFaceColor', 'b');
legend('Incremental Learning Predictions', 'Actual Values');
xlabel('Test Sample Number');
ylabel('Indicator Value');
title('Incremental Learning Prediction Results');

%% Local functions
function XCell = convertToCellArray(X)
    numSamplesLocal = size(X, 3);
    XCell = cell(numSamplesLocal, 1);
    for k = 1:numSamplesLocal
        XCell{k} = squeeze(X(:, :, k));
    end
end

function [rmse, YPred] = objective_function(params, XTrainCell, YTrain, XTestCell, YTest)
    numHiddenUnits = round(params(1));
    initialLearnRate = params(2);
    maxEpochs = round(params(3));

    if maxEpochs <= 0
        maxEpochs = 1;
    end
    if initialLearnRate <= 0
        initialLearnRate = 0.001;
    end

    layers = [
        sequenceInputLayer(size(XTrainCell{1}, 1), 'Name', 'input')
        lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(0.2, 'Name', 'dropout')
        fullyConnectedLayer(1, 'Name', 'fc')
        regressionLayer('Name', 'output')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', initialLearnRate, ...
        'MiniBatchSize', 32, ...
        'GradientThreshold', 1, ...
        'Verbose', false, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XTestCell, YTest}, ...
        'ValidationFrequency', 10, ...
        'ExecutionEnvironment', 'auto');

    net = trainNetwork(XTrainCell, YTrain, layers, options);
    YPred = predict(net, XTestCell);
    rmse = sqrt(mean((YPred - YTest).^2));
end

function [Alpha_pos, Alpha_score] = gwo(objective_function_handle, lb, ub, dim, SearchAgents_no, Max_iter, XTrainCell, YTrain, XTestCell, YTest)
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;

    Alpha_pos = zeros(1, dim);
    Alpha_score = inf;
    Beta_pos = zeros(1, dim);
    Beta_score = inf;
    Delta_pos = zeros(1, dim);
    Delta_score = inf;

    for t = 1:Max_iter
        for i = 1:SearchAgents_no
            fitness = objective_function_handle(Positions(i, :), XTrainCell, YTrain, XTestCell, YTest);

            if fitness < Alpha_score
                Alpha_score = fitness;
                Alpha_pos = Positions(i, :);
            end

            if fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness;
                Beta_pos = Positions(i, :);
            end

            if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
                Delta_score = fitness;
                Delta_pos = Positions(i, :);
            end
        end

        a = 2 - 2 * (t / Max_iter);

        for i = 1:SearchAgents_no
            for j = 1:dim
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;

                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;

                r1 = rand(); r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;

                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
        end

        fprintf('Iteration %d, Best RMSE: %.4f\n', t, Alpha_score);
    end
end
