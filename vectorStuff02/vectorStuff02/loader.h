#pragma once
#include <iostream>
#include <fstream>
using namespace std;

promptUser() {
	string userSelection;
	while (userSelection!="n" && userSelection!="c")
	{
		cout << "(n)ew net or (c)ontiue?";
		cin >> userSelection;
	}
}
network::save(const vector<vector<double>> &biases, const vector<vector<vector<double>>> &weights) {
	ofstream configuration;
	configuration.open("C:\\temp\\mnist_source\\config.csv");
	configuration.precision(16);
	configuration.fill("0");
	if (configuration.is_open())
	{
		for (int i = 0; i < &biases.size(); i++)
		{
			for (int j = 0; j < biases[i].size(); j++)
			{
				configuration << biases[i][j].tostring();
				configuration << ",";
			}
			configuration << "\n";
		}
	}
}
network::load() {


}