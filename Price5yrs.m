% Price prediction backtest using ARIMA model and 5yrs of price data

% Make sure the data has the close column
data = readtable("HistoricalData_1742929324297.csv", 'VariableNamingRule', 'preserve');

% Check if its the right data
if ~ismember('Close/Last', data.Properties.VariableNames)
    error('Missing price column. Available columns: %s', ...
          strjoin(data.Properties.VariableNames, ', '));
end

% Prep price data and calculate returns
prices = data.('Close/Last');
prices = prices(~isnan(prices) & prices > 0);  % Remove bad data
returns = price2ret(prices);

fprintf('Working with %d trading days (approx. %.1f years)\n', length(prices), length(prices)/252);

% Check if returns are stationary 
[~, pval] = adftest(returns);
if pval > 0.05
    returns = diff(returns);   % Difference if non stationary
    returns = returns(~isnan(returns));
    fprintf('Applied differencing to make stationary (ADF p=%.3f)\n', pval);
end

% Training and testing periods
trainPct = 0.7;  % 70% training
trainLen = floor(trainPct * length(returns));
testLen = length(returns) - trainLen;

trainReturns = returns(1:trainLen);
testReturns = returns(trainLen+1:end);

fprintf('\nData split:\n');
fprintf('Training: %d days\n', trainLen);
fprintf('Testing: %d days\n', testLen);

% Set up ARIMA model using (2,0,1)?
model = arima(2,0,1);  
fittedModel = estimate(model, trainReturns, 'Display', 'off');

% Run backtest
predictedDir = zeros(testLen, 1);  % Stores predictions
actualDir = sign(testReturns);      % Actual market

for day = 1:testLen
    % Update training data
    currentTrain = [trainReturns; testReturns(1:day-1)];
    
    % Make prediction
    [pred, ~] = forecast(fittedModel, 1, 'Y0', currentTrain);
    predictedDir(day) = sign(pred(end));
    
    % Re-estimate model monthly (21 days)
    if mod(day,21) == 0
        [fittedModel, ~] = estimate(model, currentTrain, 'Display', 'off');
    end
end

% Calculate performance metrics
cost = 0.0005;  % 5bps cost
stratReturns = testReturns .* predictedDir - cost * abs(predictedDir);

% Basic accuracy
accuracy = mean(predictedDir == actualDir);
confMatrix = confusionmat(actualDir, predictedDir);

% Handle potential issues
validReturns = stratReturns(~isnan(stratReturns) & ~isinf(stratReturns));
if ~isempty(validReturns)
    cumReturns = cumsum(validReturns);
    sharpe = mean(validReturns)/std(validReturns)*sqrt(252);
    
    % Risk metrics
    downside = validReturns(validReturns<0);
    sortino = mean(validReturns)/std(downside)*sqrt(252);
    
    winRate = 100*sum(validReturns>0)/length(validReturns);
    profitFactor = sum(validReturns(validReturns>0))/abs(sum(validReturns(validReturns<0)));
    maxDD = maxDrawdown(cumReturns);
else
    % If something went wrong
    cumReturns = zeros(testLen,1);
    sharpe = 0;
    sortino = 0;
    winRate = 0;
    profitFactor = 0;
    maxDD = 0;
end

% Plot results
figure('Position', [100 100 900 600]);

% Plot cumulative returns 
subplot(2,1,1);
plot(cumReturns, 'LineWidth', 2);
hold on;
plot(cumsum(testReturns), 'k--');
title('Strategy vs Buy-and-Hold');
legend('Our Strategy (with costs)', 'Buy-and-Hold', 'Location', 'northwest');
grid on;

% Accuracy over time
subplot(2,1,2);
plot(movmean(predictedDir==actualDir, 21), 'LineWidth', 2);
yline(0.5, 'r--');  % 50% accuracy line
title('Rolling 21-Day Accuracy');
ylim([0 1]);
grid on;

% Performance summary
fprintf('\n=== Performance Summary ===\n');
fprintf('Direction Accuracy: %.1f%%\n', accuracy*100);
fprintf('Annual Sharpe: %.2f\n', sharpe);
fprintf('Sortino Ratio: %.2f\n', sortino);
fprintf('Win Rate: %.1f%%\n', winRate);
fprintf('Profit Factor: %.2f\n', profitFactor);
fprintf('Max Drawdown: %.2f%%\n', 100*maxDD);
fprintf('\nConfusion Matrix:\n');
disp(confMatrix);

% Calculate max drawdown
function dd = maxDrawdown(cumReturns)
    runningMax = cummax(cumReturns);
    drawdowns = (runningMax - cumReturns) ./ runningMax;
    drawdowns(runningMax == 0) = 0;  % Handle division by zero
    dd = max(drawdowns);
    if isempty(dd) || isnan(dd)
        dd = 0;
    end
end
