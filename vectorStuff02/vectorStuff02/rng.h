#pragma once
#include <random>
#include <iostream>
#include <math.h>
/*
double rng() {
double lower_bound = 0;
double upper_bound = 1;
std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
std::default_random_engine re;
double randomDouble = unif(re);
return randomDouble;
}
*/
double rngDouble(int dev) {
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_real_distribution<double> uni(-(1 / sqrt(dev)), (1/sqrt(dev)));
	double x = uni(rng);//((double)rand() / (RAND_MAX));
	return x;
}
int rngInt(int min, int max) {
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni(min, max); // guaranteed unbiased
	auto random_integer = uni(rng);
	//int x = ((int)rand() / (RAND_MAX));
	//std::cout << x << endl;
	return random_integer;
}