// vectorStuff.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "rng.h"
#include "mnist_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include <math.h>
using namespace std;

class network
{
private:
	vector<int> layerSizes;
	vector<vector<double>> biases;
	vector<double> weights;
public:
	vector<vector<double>> generateBiases(vector<int>);
	//vector<double> generateWeightsPerLayer(int);
	void setLayerSizes(vector<int> inputSizes) { layerSizes = inputSizes; }
	void setBiases(vector<int> inputSize) { biases = generateBiases(inputSize); }
	vector<vector<vector<double>>> setWeights(vector<int>);
	vector<double> getLayerBiases(vector<vector<double>> &biases, int layerNumber);
	vector<double> getNeuronWeights(vector<vector<vector<double>>> &weights, int layerNumber, int neuronNumber);

	void print()
	{
		for (int i = 0; i < layerSizes.size(); i++)
		{
			cout << "Layer " << i << ": " << layerSizes[i]<<"\n";
		}
	}
	void print(vector<double> dVec) {
		cout << "\n";
		for (int i = 0; i < dVec.size(); i++)
		{
			cout << "Element " << i << ": " << dVec[i] << ", ";
		}
	}	
	void print(vector<vector<double>> tupleVec) {
		cout << "\n";
		for (int i = 0; i < tupleVec.size(); i++)
		{
			cout << "\n";
			for (int j = 0; j < tupleVec[i].size(); j++)
			{
				cout <<tupleVec[i][j] << ", ";
			}
			
		}
		/*
		for (int i = 0; i < tupleVec.size(); i++)
		{
			for (int j = 0; j < tupleVec[i].size(); j++)
			{
				cout << "Layer " << i << ", " << "Element " << j << ": " << tupleVec[i][j] << "\n";
			}
			cout << "\n";
		}*/
	}
	void print(vector<vector<vector<double>>> tripleVec) {
		cout << "\n";
		for (int i = 0; i < tripleVec.size(); i++)
		{
			cout << "Layer " << i << ": \n";
			for (int j = 0; j < tripleVec[i].size(); j++)
			{
				cout << "Node " << j << ": \n";
				for (int k = 0; k < tripleVec[i][j].size(); k++)
				{
					cout << tripleVec[i][j][k] << ", ";
				}
				cout << "\n";
			}

		}
		/*
		for (int i = 0; i < tripleVec.size(); i++)
		{
			cout << "Layer " << i << ": \n";
			for (int j = 0; j < tripleVec[i].size(); j++)
			{
				cout << "Node " << j << ": \n";
				for (int k = 0; k < tripleVec[i][j].size(); k++)
				{
					cout <<"Weight "<<k<<": " << tripleVec[i][j][k] << "\n";
				}
				cout << "\n";
			}

		}*/
	}
	void feedforward(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, vector<vector<double>> &inActivations, vector<vector<double>> &weightedInputs, vector<vector<double>> &outActivations);
	void backdrop(vector<double> &activationsFromPreviousLayer, vector<double> &desiredOutput, vector<vector<vector<double>>> &inWeights, vector<vector<double>> &inBiases, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights);
	vector<vector<double>> vectorizeLabels(const vector<double> &labels);
	void updateMiniBatch(const vector<vector<double>> &miniBatch, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRat);
	//void generateMiniBatch(int miniBatchSize, const vector<vector<vector<double>>> &inputData, vector<vector<vector<double>>> &outputBatch);
	void stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, int epochs, int miniBatchSize, int learningRate, const vector<vector<vector<double>>> &testData, vector<vector<vector<double>>>&weights, vector<vector<double>> &biases);
	double getActivation(vector<double> &input, const vector<double> &weights, const double bias);
};

vector<double> generateRandomVec(int length) {

	vector<double> randomVec;
	for (int j = 0; j < length; j++)
	{
		randomVec.push_back(rngDouble());
	}
	return randomVec;
}
vector<vector<double>> network::generateBiases(vector<int> networkDimensions)
{
	vector<vector<double>> biases;
	vector<double> biasesInputLayer;
	for (int i = 0; i < networkDimensions[0]; i++)
	{
		
		biasesInputLayer.push_back(0);
	}
	biases.push_back(biasesInputLayer);
	for (int i = 1; i < networkDimensions.size(); i++)
	{
		vector<double> biasesPerLayer;
		biasesPerLayer = generateRandomVec(networkDimensions[i]);
		biases.push_back(biasesPerLayer);
	}

	return biases;
}
/*vector<double> network::generateWeightsPerLayer(int layerSize) {
	vector<double> weightsPerLayer;
	weightsPerLayer = generateRandomVec(layerSize);
	return weightsPerLayer;
}*/
vector<vector<vector<double>>> network::setWeights(vector<int> inputSize)
{
	vector<vector<vector<double>>> weights;
	for (int i = 0; i < inputSize.size()-1; i++)	//for each layer
	{

		vector<vector<double>> weightsPerNeuron;
		for (int j = 0; j < inputSize[i+1]; j++)
		{			
			weightsPerNeuron.push_back(generateRandomVec(inputSize[i]));
		}
		weights.push_back(weightsPerNeuron);
	}
	return weights;
}
vector<double> network::getLayerBiases(vector<vector<double>> &biases, int layerNumber) {
	return biases[layerNumber];
}
vector<double> network::getNeuronWeights(vector<vector<vector<double>>> &weights, int layerNumber, int neuronNumber)
{
	return weights[layerNumber][neuronNumber];
}
vector<vector<double>> network::vectorizeLabels(const vector<double> &labels) {
	vector<vector<double>> vectorizedLabels;
	for (int i = 0; i < labels.size(); i++)
	{
		vector<double> label;
		for (int j = 0; j < 10; j++)
		{
			label.push_back(0);					
		}
		label[labels[i]] = 1;
		vectorizedLabels.push_back(label);
	}
	return vectorizedLabels;
}
int getIntInput() {
	while (1) // Loop until user enters a valid input
	{
		cout << "Enter number of neurons per Layer or 0 when done: ";
		int x;
		cin >> x;

		if (cin.fail()) // has a previous extraction failed?
		{
			// yep, so let's handle the failure
			cin.clear(); // put us back in 'normal' operation mode
			cin.ignore(32767, '\n'); // and remove the bad input
		}
		else
		{
			return x;// nope, so return our good x
		}
	}
}
vector<int> getNetworkSize() 
{
	vector<int> networkSize;
	int userInput;
	userInput = getIntInput();

	if (userInput == 0)
	{
		return networkSize;
	}
	else
	{
		while (userInput!=0)
		{
			networkSize.push_back(userInput);
			userInput = getIntInput();
		}
		return networkSize;
	}	
}
double sigmoid(double z)
{
	double x;
	x = (1 + (exp((-1) * z)));
	return (1 / x);
}
double sigmoidPrime(double z) {
	return (sigmoid(z) * (1 - sigmoid(z)));
}
vector<double> costDerivative(const vector<double> &outputActivations, const vector<double> &desiredOutput) {
	vector<double> errors;
	for (int i = 0; i < outputActivations.size(); i++)
	{
		errors.push_back(outputActivations[i]-desiredOutput[i]);  //to be tested
	}
	return errors;
}
void zip (vector<double> &a, vector<double> &b, vector<vector<double>> &outVec) 
{
	if (a.size() >= b.size())	{
		for (int i = 0; i < b.size(); i++)
		{
			vector<double> temp;
			temp.push_back(a[i]);
			temp.push_back(b[i]);
			outVec.push_back(temp);
		}

	}
	else {
		for (int i = 0; i < a.size(); i++)
		{
			vector<double> temp;
			temp.push_back(a[i]);
			temp.push_back(b[i]);
			outVec.push_back(temp);
		}
	}
	
}
void zip (vector<vector<double>> &a, vector<double> &b, vector<vector<vector<double>>> &outVec) 
{
	if (a.size() >= b.size()) {
		for (int i = 0; i < b.size(); i++)
		{
			vector<double>bElement;
			vector<vector<double>> temp;
			bElement.push_back(b[i]);
			temp.push_back(a[i]);
			temp.push_back(bElement);
			outVec.push_back(temp);
		}

	}
	else {
		for (int i = 0; i < a.size(); i++)
		{
			vector<double>bElement;
			vector<vector<double>> temp;
			bElement.push_back(b[i]);
			temp.push_back(a[i]);
			temp.push_back(bElement);
			outVec.push_back(temp);
		}
	}

}
void zip(vector<vector<double>> &a, vector<vector<double>> &b, vector<vector<vector<double>>> &outVec) //to be tested
{
	if (a.size() >= b.size()) {
		for (int i = 0; i < b.size(); i++)
		{
			vector<double>aElement;
			vector<double>bElement;
			vector<vector<double>> tempElement;
			vector<vector<vector<double>>> temp;
			aElement = a[i];
			bElement = b[i];
			tempElement.push_back(aElement);
			tempElement.push_back(bElement);
			temp.push_back(tempElement);
			
			outVec.push_back(tempElement);
		}

	}
	else {
		for (int i = 0; i < a.size(); i++)
		{
			vector<double>aElement;
			vector<double>bElement;
			vector<vector<double>> tempElement;
			vector<vector<vector<double>>> temp;
			aElement = a[i];
			bElement = b[i];
			tempElement.push_back(aElement);
			tempElement.push_back(bElement);
			temp.push_back(tempElement);

			outVec = temp;
		}
	}

}
vector<double> dotProduct(const vector<double> &a, const vector<double> &b) {
	vector<double> results;
	for (int i = 0; i < a.size(); i++)
	{
		results.push_back((a[i] * b[i]));
	}
	return results;
}
double sumOfDotProduct(const vector<double> &a, const vector<double> &b)
{
	double x = 0;	
	for (int i = 0; i < a.size(); i++)
	{
		x = x + (a[i] * b[i]);		
	}
	return x;
}
void fillVecWithZeroes(vector<double> &inVec) {
	for each (double element in inVec)
	{
		element = 0;
	}
}
void network::backdrop(vector<double> &activationsFromPreviousLayer,vector<double> &desiredOutput, vector<vector<vector<double>>> &inWeights, vector<vector<double>> &inBiases, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights) //Return a tuple "(nabla_b, nabla_w)" representing the gradient for the cost function C_x.
{
	vector<vector<double>> biasesUpdated;
	vector<vector<vector<double>>> weightsUpdated;
	vector<vector<double>> weightedInputs; //"zs"
	vector<vector<double>> listOfActivations;
	vector<vector<double>> calculatedActivations;
	//vector<vector<double>> blub;
	listOfActivations.push_back(activationsFromPreviousLayer);
	vector<double> sigmoidPrimeOfZ;
	//feedforward()
	
	feedforward(inWeights, inBiases, listOfActivations, weightedInputs, calculatedActivations);
	//backward pass
	//secondToLast = myVector[myVector.size() - 2];

	//get SigmoidPrime of Zs via Zs from last Layer to calculate derivative of costs ()
	//for each (vector<double> layerActivation in calculatedActivations)
	for (int i = 1; i < calculatedActivations.size(); i++) {
		listOfActivations.push_back(calculatedActivations[i]);
	}
	
	for (int i = 0; i < weightedInputs[weightedInputs.size()-2].size(); i++) 
	{
		sigmoidPrimeOfZ.push_back(sigmoidPrime(weightedInputs[weightedInputs.size() - 2][i]));
	}
	vector<vector<double>> delta;
	//move backwards, compare activations to desiredOutputs

	//get costs in last layer
	vector<double> costsLastLayer;
	costsLastLayer = costDerivative(listOfActivations[listOfActivations.size()-1], desiredOutput);
	
	//delta = costDerivative(listOfActivations[listOfActivations.size()-2], desiredOutput) * "sigmoidPrime(weightedInputs)";
	
	for (int i = 0; i < listOfActivations.size(); i++)
	{
		vector<double> deltaPerLayer;
		for (int j = 0; j < weightedInputs[weightedInputs.size() - 2].size(); j++)
		{
			deltaPerLayer.push_back(sumOfDotProduct(costDerivative(listOfActivations[listOfActivations.size() - 2], desiredOutput), sigmoidPrimeOfZ));//ladida
		}
		delta.push_back(deltaPerLayer);
	}
	
	print(delta);
	nabla_biases = delta;
	vector<vector<double>>costs;
	

	
}
void network::updateMiniBatch(const vector<vector<double>> &dataSet, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate)
{
	vector<vector<double>> nabla_biases(biases);
	vector<vector<vector<double>>> nabla_weights(weights);
	for (int i = 0; i < nabla_biases.size(); i++)
	{
		fillVecWithZeroes(nabla_biases[i]);
	}
	for (int k = 0; k < nabla_weights.size(); k++)
	{
		for (int j = 0;  j< weights[k].size(); j++)
		{
			fillVecWithZeroes(nabla_weights[k][j]);
		}
	}
	
//	for (int i = 0; i < dataSet.size(); i++)
	{
		//vectors delta__nabla_weights, delta_nabla_biases = output of backdrop
		//vector<double> x;
		vector<double>activationsAtInputLayer(dataSet[0]);
		vector<double>output(dataSet[1]);
		network::backdrop(activationsAtInputLayer, output, weights, biases, nabla_biases, nabla_weights);

	}
	//ladida3
}
void generateMiniBatch(int miniBatchSize, const vector<vector<vector<double>>> &inputData, vector<vector<vector<double>>> &outputBatch){
	for (int i = 0; i < miniBatchSize; i++)
	{
		outputBatch.push_back(inputData[rngInt()]);//pick random datasets from input
	}
}
void network::stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, int epochs, int miniBatchSize, int learningRate, const vector<vector<vector<double>>> &testData, vector<vector<vector<double>>>&weights, vector<vector<double>> &biases) {
	//example call in python >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data) 
	for (int i = 0; i <= epochs; i++)
	{
		vector<vector<vector<double>>> miniBatch;
		generateMiniBatch(miniBatchSize, trainingData, miniBatch);
		for each (vector<vector<double>> dataSet in miniBatch)
		{
			
			
			updateMiniBatch(dataSet, biases,weights,learningRate);
			//updateMiniBatch(vector<vector<double>> &dataSet, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate)
			//ladida
		}
	}

}
void network::feedforward(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, vector<vector<double>> &inActivations, vector<vector<double>> &weightedInputs, vector<vector<double>> &outActivations)
{
	vector<double> weightedLayerInputs;
	weightedInputs.push_back(inActivations[0]);
	outActivations.push_back(inActivations[0]);
	for (int i = 0; i < inWeights.size(); i++)
	{
		vector<double> layerActivations;
		for (int j = 0; j < inWeights[i].size(); j++)
		{						
			vector<double> z;
			layerActivations.push_back(getActivation(outActivations[i], inWeights[i][j], inBiases[i][j]));				
			
			
			z = dotProduct( inWeights[i][j], outActivations[i]) ; //0 is falsch
			weightedLayerInputs = z;
			vector<double> sigmoids;
			/*for each (double weightedInput in weightedLayerInputs)
			{
				double y = 0;
				y = sigmoid(weightedInput);
				sigmoids.push_back(y);
			}*/
	
			//layerActivations.push_back(sigmoids[i] + (inBiases[i][j]));
			/*if (i==0)//first layer has no biases
			{
				layerActivations.push_back(y);
			}
			else
			{
				layerActivations.push_back(y+inBiases[i][j]);
			}*/
			//z = zip()[i][i];
			
		}
		weightedInputs.push_back(weightedLayerInputs);
		outActivations.push_back(layerActivations);
	}	
}
double network::getActivation(vector<double> &input, const vector<double> &weights, const double bias) 
{
	vector<double>outActivations;
	double activation=0;
	for (int i = 0; i < input.size(); i++)
	{
		outActivations.push_back((input[i] * weights[i] + bias));
	}
	for (int j = 0; j < outActivations.size(); j++)
	{
		activation = activation + outActivations[j];
	}
	return activation;
}

int main()
{
	int epochs = 5;
	int miniBatchSize = 10;
	int learningRate = 3;
	//vector<double> hyperParam{ epochs, miniBatchSize, learningRate };

	vector<vector<double>> allImageData;
	ReadMNISTImages(10000, 784, allImageData);
	vector<double> labels;
	ReadMNISTLabels(10000, labels);
	vector<int> networkSize;
	networkSize = getNetworkSize();
	network net{  };
	net.setLayerSizes(networkSize);
	//net.print();
	vector<vector<double>> biases;
	biases = net.generateBiases(networkSize);
	cout << "BIASES###";
	//net.print(biases);
	vector<vector<vector<double>>> weights;
	weights = net.setWeights(networkSize);
	cout << "WEIGHTS ###";
	//net.print(weights);
	vector<vector<vector<double>>> trainingData;
	vector<vector<double>> vectorizedLabels;
	vectorizedLabels = net.vectorizeLabels(labels);
	zip(allImageData, vectorizedLabels, trainingData);
	//net.print(trainingData);
	vector<vector<vector<double>>> sample;
	//net.print(trainingData);
	net.stochasticGradientDescent(trainingData, epochs, miniBatchSize,learningRate,trainingData, weights, biases);
	//stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, int epochs, int miniBatchSize, int learningRate, const vector<vector<vector<double>>> &testData)
	//net.print(net.vectorizeLabels(labels));
	//net.print(trainingData);
	//zip(biases[0], weights[0][0], zipped);
	//net.print(zipped);
	//net.print(allImageData);
	//net.print(labels);
	//double test = rng();
	cout << "feeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed";
	vector<vector<double>> out;
	//out = net.feedforward(weights, biases, allImageData);
	//net.print(out);
	//net.print(zipped);
	//net.print(miniBatches);
	string blub;
	cin >> blub;
	return 0;
}