clear
clc
close all

data = csvread('wmoutput.csv');
x = data(:, 1);
tau_result = data(:, 2);
tau_expected = data(:, 3);

figure
hold on
plot(x, tau_expected, 'LineWidth', 2)
plot(x, tau_result, 'LineWidth', 2)