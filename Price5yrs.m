% Load dataset (update filename if needed)
data = readtable("HistoricalData_1742929324297.csv", 'VariableNamingRule', 'preserve');

% Verify data
if ~ismember('Close/Last', data.Properties.VariableNames)
    error('Price column not found. Available columns: %s', ...
          strjoin(data.Properties.VariableNames, ', '));
end

prices = data.('Close/Last');
prices = prices(~isnan(prices) & prices > 0);
returns = price2ret(prices);

fprintf('Loaded %d trading days (%.1f years)\n', length(prices), length(prices)/252);

%% Step 2: Stationarity Handling
[~, pval] = adftest(returns);
if pval > 0.05
    returns = diff(returns);
    returns = returns(~isnan(returns));
    fprintf('Applied differencing (p=%.3f)\n', pval);
end

%% Step 3: Train-Test Split
train_ratio = 0.7;
train_size = floor(train_ratio * length(returns));
test_size = length(returns) - train_size;

train_data = returns(1:train_size);
test_data = returns(train_size+1:end);

fprintf('\nTrain-Test Split:\n');
fprintf('-> Training: %d days\n', train_size);
fprintf('-> Testing: %d days\n', test_size);

%% Step 4: Reliable Model Setup
model = arima(2,0,1);  % Basic but stable model
fit = estimate(model, train_data, 'Display', 'off');

%% Step 5: Guaranteed Prediction Backtest
predicted_directions = zeros(test_size, 1);
actual_directions = sign(test_data);

for t = 1:test_size
    current_train = [train_data; test_data(1:t-1)];
    
    % Always make a prediction (no volatility filter)
    [Y, ~] = forecast(fit, 1, 'Y0', current_train);
    predicted_directions(t) = sign(Y(end));
    
    % Monthly re-estimation
    if mod(t,21) == 0
        [fit, ~] = estimate(model, current_train, 'Display', 'off');
    end
end

%% Step 6: Robust Performance Calculation
% Basic metrics
accuracy = mean(predicted_directions == actual_directions);
conf_mat = confusionmat(actual_directions, predicted_directions);

% Strategy returns with 5bps transaction cost
transaction_cost = 0.0005;
strategy_returns = test_data .* predicted_directions - transaction_cost * abs(predicted_directions);

% Safe metrics calculation
valid_returns = strategy_returns(~isnan(strategy_returns) & ~isinf(strategy_returns));
if ~isempty(valid_returns)
    cum_returns = cumsum(valid_returns);
    sharpe_ratio = mean(valid_returns)/std(valid_returns)*sqrt(252);
    sortino_ratio = mean(valid_returns)/std(valid_returns(valid_returns<0))*sqrt(252);
    win_rate = 100*sum(valid_returns>0)/length(valid_returns);
    profit_factor = sum(valid_returns(valid_returns>0))/abs(sum(valid_returns(valid_returns<0)));
    max_dd = my_maxdrawdown(cum_returns);  % Using our custom function
else
    cum_returns = zeros(test_size,1);
    sharpe_ratio = 0;
    sortino_ratio = 0;
    win_rate = 0;
    profit_factor = 0;
    max_dd = 0;
end

%% Step 7: Visualization
figure('Position', [100 100 900 600]);

% Cumulative returns
subplot(2,1,1);
plot(cum_returns, 'LineWidth', 2);
hold on;
plot(cumsum(test_data), 'k--');
title('Strategy Performance');
legend('Strategy (5bps costs)', 'Buy-and-Hold', 'Location', 'northwest');
grid on;

% Prediction accuracy
subplot(2,1,2);
plot(movmean(predicted_directions==actual_directions, 21), 'LineWidth', 2);
yline(0.5, 'r--');
title('21-Day Moving Accuracy');
ylim([0 1]);
grid on;

%% Step 8: Performance Report
fprintf('\n=== Strategy Performance ===\n');
fprintf('Direction Accuracy: %.1f%%\n', accuracy*100);
fprintf('Annualized Sharpe: %.2f\n', sharpe_ratio);
fprintf('Sortino Ratio: %.2f\n', sortino_ratio);
fprintf('Win Rate: %.1f%%\n', win_rate);
fprintf('Profit Factor: %.2f\n', profit_factor);
fprintf('Max Drawdown: %.2f%%\n', 100*max_dd);
fprintf('\nConfusion Matrix:\n');
disp(conf_mat);

%% Custom Max Drawdown Function
function max_dd = my_maxdrawdown(cum_returns)
    % Calculate running maximum
    running_max = cummax(cum_returns);
    
    % Calculate drawdowns
    drawdowns = (running_max - cum_returns) ./ running_max;
    
    % Handle case where running_max is zero
    drawdowns(running_max == 0) = 0;
    
    % Maximum drawdown
    max_dd = max(drawdowns);
    
    % If no drawdown occurred
    if isempty(max_dd) || isnan(max_dd)
        max_dd = 0;
    end
end