#pragma once
#include <random>
#include <iostream>
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
double rngDouble() {
	double x = ((double)rand() / (RAND_MAX));
	return x;
}
int rngInt() {
	int x = ((int)rand() / (RAND_MAX));
	return x;
}