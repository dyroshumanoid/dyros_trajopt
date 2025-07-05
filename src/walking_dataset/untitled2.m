 clc
clear all
close all

data = readmatrix('tocabi_walking.txt');

size(data)

0.6 / 0.005
0.3 / 0.005

figure()
DOF = 0
plot(data(:,[DOF+1:DOF+12]))
figure()
DOF = 33
plot(data(:,[DOF+1:DOF+12]))
figure()
DOF = 66
plot(data(:,[DOF+1:DOF+12]))
