// vectorStuff.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "rng.h"
#include "mnist_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <algorithm>
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
			cout << dVec[i]<<", ";
		//	cout << "Element " << i << ": " << dVec[i] << ", ";
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
	void backdrop(const int &miniBatchSize,const int &learningRate, vector<double> &activationsFromPreviousLayer, vector<double> &desiredOutput, vector<vector<vector<double>>> &inWeights, vector<vector<double>> &inBiases, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights);
	vector<vector<double>> vectorizeLabels(const vector<double> &labels);
	void updateMiniBatch(const vector<vector<double>> &miniBatch, const int &miniBatchSize, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRat, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights, vector<vector<double>> &delta_nabla_biases, vector<vector<vector<double>>> &delta_nabla_weights);
	//void generateMiniBatch(int miniBatchSize, const vector<vector<vector<double>>> &inputData, vector<vector<vector<double>>> &outputBatch);
	void stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, double epochs, double miniBatchSize, double learningRate, const vector<vector<vector<double>>> &testData, vector<vector<vector<double>>>&weights, vector<vector<double>> &biases);
	double weightInput(vector<double> &input, const vector<double> &weights, const double &bias);
	void getDesiredOutput(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<double> &desiredOutput, vector<vector<double>> &listOfDesiredOutputs);
	void getErrors(const vector<vector<double>> &realOutputs, const vector<vector<double>> &listOfDesiredOutputs, vector<vector<double>> &errors);
	void getSigmoidPrime(const vector<vector<double>> &weightedInputs, vector<vector<double>> &sigmoidPrimeofZ);
	void getDelta(const vector<vector<double>> &costs, const vector<vector<double>> &sigmoidPrimeOfZ, vector<vector<double>> &delta);
	void getNablaWeights(const vector<vector<double>> &delta, const vector<vector<double>> &listOfActivations, vector<vector<vector<double>>> &nabla_weights);
	void evaluate(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<vector<vector<double>>> &inputData);
	

};

vector<double> generateRandomVec(int length) {

	vector<double> randomVec;
	for (int j = 0; j < length; j++)
	{
		randomVec.push_back(rngDouble(length));
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
		//biasesPerLayer = generateRandomVec(networkDimensions[i]);
		vector<double> randomVec;
		for (int j = 0; j < networkDimensions[i]; j++)
		{
			randomVec.push_back(rngDouble(1));
		}
		biases.push_back(randomVec);
	}

	return biases;
}
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
	//double blub;
	//blub = (1.0 / (1.0 + (exp(-z))));
	return (1.0 / (1.0 + (exp(-1*z))));
	/*
	double x;
	x = (1 + (exp((-1) * z)));
	return (1 / x);*/
}
double sigmoidPrime(double z) {
	//cout << (sigmoid(z) * (1.0 - (sigmoid(z))));
	return (sigmoid(z) * (1.0 - (sigmoid(z))));
}
vector<double> costDerivative(const vector<double> &outputActivations, const vector<double> &desiredOutput) //parameters: realOutput, desiredOutput
{
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
	vector<double> sumVec;
	sumVec = dotProduct(a, b);
	for (int i = 0; i < sumVec.size(); i++)
	{
		x += sumVec[i];		
	}
	return x;
}
double sumOfVec(const vector<double> &a)
{
	double x = 0;
	
	for (int i = 0; i < a.size(); i++)
	{
		x += a[i];
	}
	return x;
}
int posOfHighestElement(const vector<double> &a) {
	auto highestElement = max_element(begin(a), end(a));
	auto posOfHighestElementInVector = distance(begin(a), highestElement);
	return posOfHighestElementInVector;
}
bool hasSameHighestElement(const vector<double> &a, const vector<double> &b){
	if (posOfHighestElement(a) == posOfHighestElement(b))
	{
		return true;
	}
	return false;
}
vector<double> fillVecWithZeroes(vector<double> &inVec) {
	vector<double> outVec;
	for (int i=0; i< inVec.size(); i++)
	{
		outVec.push_back(0);
	}
	return outVec;
}
vector<double> sumPerElement(const vector<double> &a, const vector<double> &b) {
	vector<double> outVec;
	for (int i = 0; i < a.size(); i++)
	{
		outVec.push_back(a[i]+b[i]);
	}
	return outVec;
}
void network::backdrop(const int &miniBatchSize, const int &learningRate, vector<double> &activationsFromPreviousLayer,vector<double> &desiredOutput, vector<vector<vector<double>>> &inWeights, vector<vector<double>> &inBiases, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights) //Return a tuple "(nabla_b, nabla_w)" representing the gradient for the cost function C_x.
{
	vector<vector<double>> biasesUpdated;
	vector<vector<vector<double>>> weightsUpdated;
	vector<vector<double>> weightedInputs; //"zs"
	vector<vector<double>> listOfActivations;
	vector<vector<double>> calculatedActivations;
	vector<vector<double>> sigmoidPrimeOfZ;
	vector<vector<double>> costs; //costs, errors, wuteva
	vector<vector<double>> listOfDesiredOutputs;
	vector<vector<double>> delta;
	
	//vector<vector<double>> blub;
	listOfActivations.push_back(activationsFromPreviousLayer);
	
	//feedforward()

	feedforward(inWeights, inBiases, listOfActivations, weightedInputs, calculatedActivations); //calculates activations
	//listOfActivations[0] = calculatedActivations[0];
	for (int i = 1; i < calculatedActivations.size(); i++) {
		//listOfActivations[i] = calculatedActivations[i];
		listOfActivations.push_back(calculatedActivations[i]);
	}
	//***backward pass***
	//getDesiredOutput(inWeights, inBiases, desiredOutput, listOfDesiredOutputs);
	//get costs
	//getErrors(listOfActivations, listOfDesiredOutputs, costs);// compare activations to desiredOutputs
	//getErrors(calculatedActivations, listOfDesiredOutputs, costs);// compare activations to desiredOutputs
	//secondToLast = myVector[myVector.size() - 2];

	//get SigmoidPrime of Zs via Zs from last Layer to calculate derivative of costs ()
	//for each (vector<double> layerActivation in calculatedActivations)
	getSigmoidPrime(weightedInputs, sigmoidPrimeOfZ); 

	
	vector<double> deltaLastLayer;
	for (int i = 0; i < desiredOutput.size(); i++)
	{
		double nodeDelta;
		
		//nodeDelta = (listOfActivations[listOfActivations.size() - 1][i] - desiredOutput[i]) * sigmoidPrimeOfZ[sigmoidPrimeOfZ.size()-1][i];
		nodeDelta = (listOfActivations[listOfActivations.size() - 1][i] - desiredOutput[i]) * sigmoidPrimeOfZ[sigmoidPrimeOfZ.size() - 1][i];
		deltaLastLayer.push_back(nodeDelta);
	}
	delta.push_back(deltaLastLayer);
	//transpose Weights
	vector<vector<vector<double>>>weightsTransposed;
	vector<double> deltaOtherLayer;
	for (int i = calculatedActivations.size(); i > 1; i--)
	{	
		vector<vector<double>> layerWeightsTransposed;
		for (int j = 0; j < calculatedActivations[i-2].size(); j++)
		{
			vector<double> weightTranspose;
			for (int k = 0; k < calculatedActivations[i-1].size(); k++)
			{
				weightTranspose.push_back(inWeights[i-2][k][j]);
			}
			layerWeightsTransposed.push_back(weightTranspose);
		}		
		weightsTransposed.push_back(layerWeightsTransposed);
	}
	//get delta up to last layer
	//delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
	for (int i = 0; i < listOfActivations.size()-1; i++)
	{
		vector<double> layerDelta;
		for (int j = 0; j < listOfActivations[listOfActivations.size()-(i+2)].size(); j++)
		{
			/*
			for (int k = 0; k < weightsTransposed[i][j].size(); k++)
			{
				double blub;
				blub = weightsTransposed[i][j][k] * delta[delta.size() - (i + 1)][k] * sigmoidPrimeOfZ[sigmoidPrimeOfZ.size() - (i + 1)][k];
				layerDelta.push_back(blub);
			}
			delta.insert(delta.begin(), layerDelta);*/
			
			double dp = 0;
			dp = sumOfDotProduct(weightsTransposed[i][j],delta[delta.size()-(i+1)]);
			layerDelta.push_back(dp* sigmoidPrimeOfZ[sigmoidPrimeOfZ.size() - (i + 2)][j]);
			
		}
		delta.insert(delta.begin(), layerDelta);
	}
	/*
	for (int i = 0; i< calculatedActivations.size()-1; i++)
	{
		vector<double> layerDelta;
		for (int j = 0; j < calculatedActivations[calculatedActivations.size()-(i+2)].size();  j++)
		{
			double nodeDelta;
			nodeDelta = sumOfDotProduct(weightsTransposed[i][j], delta[delta.size() - (i+1)]) * sigmoidPrimeOfZ[sigmoidPrimeOfZ.size()-(i+2)][j];
			layerDelta.push_back(nodeDelta);
		}
		delta.insert(delta.begin(), layerDelta);
	}
	*/
	//nodeDelta = sumOfDotProduct(weightTranspose, delta[i])*sigmoidPrimeOfZ[1];
	
	//deltaOtherLayer = sumOfDotProduct(weights, delta[]);
	//delta= costs[vorletztes layer] * sigmoidPrimeOfZ[vorletztes layer]
	/*
	vector<double> deltaLastLayer;
	for (int i = 0; i < costs[costs.size()-1].size(); i++)
	{
		deltaLastLayer.push_back(costs[costs.size() - 1][i] * sigmoidPrimeOfZ[sigmoidPrimeOfZ.size()-1][i]);
	}
	delta.push_back(deltaLastLayer);
	*/
	/*
	vector<double>deltaLastLayer;
	for (int i = 0; i < desiredOutput.size(); i++)
	{
		deltaLastLayer.push_back( (calculatedActivations[calculatedActivations.size() - 1][i] - desiredOutput[i]) * sigmoidPrime(weightedInputs[weightedInputs.size() - 1][i]) );
	}
	delta.push_back( deltaLastLayer);*/
	//getDelta(costs, sigmoidPrimeOfZ, delta);

	//nabla_biases[nabla_biases.size() - 2] = delta[0];
	/*for (int i = 0; i < nabla_biases.size(); i++)
	{
		nabla_biases[i] = delta[i];
	}*/
	nabla_biases = delta;
	getNablaWeights(delta, calculatedActivations, nabla_weights);//broken?
	//nabla_weights[nabla_weights.size() - 2] = dotProduct(delta, listOfActivations[listOfActivations.size()-2]);
	/*
	//edit biases
	for (int i = 0; i < new_biases.size(); i++)
	{
		new_biases[i] = sumPerElement(nabla_biases[i], inBiases[i]);
	}
	//edit weights
	for (int i = 0; i < new_weights.size(); i++)
	{
		for (int j = 0; j<nabla_weights[i].size(); j++)
		{
			new_weights[i][j] = sumPerElement(nabla_weights[i][j], inWeights[i][j]);
		}
	}
	//edit biases
	for (int i = 0; i < new_biases.size(); i++)
	{
		for (int j = 0; j < new_biases[i].size(); j++)
		{
			new_biases[i][j] = ((new_biases[i][j]-(learningRate/miniBatchSize))*nabla_biases[i][j]);//w-(eta/length)*nablaW
		}
	
	}
	//edit weights
	for (int i = 0; i < new_weights.size(); i++)
	{
		for (int j = 0; j<nabla_weights[i].size(); j++)
		{
			for (int k =0; k<nabla_weights[i][j].size(); k++)
			{
				new_weights[i][j][k] = ((new_weights[i][j][k] - (learningRate / miniBatchSize))*nabla_weights[i][j][k]);
			}			
		}
	}*/
}
void network::updateMiniBatch(const vector<vector<double>> &dataSet, const int &miniBatchSize, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights, vector<vector<double>> &delta_nabla_biases, vector<vector<vector<double>>> &delta_nabla_weights)
{
	
	//for (int ladida = 0; ladida < miniBatchSize; ladida++) //muss noch minibatch_size werden
	{

		vector<vector<double>> new_biases;
		vector<vector<vector<double>>> new_weights;
		//vectors delta__nabla_weights, delta_nabla_biases = output of backdrop
		//vector<double> x;
		vector<double>activationsAtInputLayer(dataSet[0]);
		vector<double>output(dataSet[1]);
		network::backdrop(miniBatchSize, learningRate, activationsAtInputLayer, output, weights, biases, nabla_biases, nabla_weights);
		new_biases = nabla_biases;
		new_weights = nabla_weights;
		//edit nabla_biases
		
		for (int i = 1; i < biases.size(); i++)
		{
			for (int j = 0; j < biases[i].size(); j++)
			{
				delta_nabla_biases[i][j] = nabla_biases[i][j]+ delta_nabla_biases[i][j];
			}
		}
		//edit nabla_weights
		for (int i = 0; i < weights.size(); i++)
		{
			for (int j = 0; j < weights[i].size(); j++)
			{
				for (int k = 0; k < weights[i][j].size(); k++)
				{
					delta_nabla_weights[i][j][k] = nabla_weights[i][j][k] + delta_nabla_weights[i][j][k];
				}
				
			}
			
		}		
		

	}
	
}
void generateMiniBatch(int miniBatchSize, const vector<vector<vector<double>>> &inputData, vector<vector<vector<double>>> &outputBatch){
	for (int i = 0; i < miniBatchSize; i++)
	{
		int j = 0;
		j = rngInt(0,inputData.size()-1);
		/*
		cout << posOfHighestElement(inputData[j][1]) << endl;
		if (hasSameHighestElement(inputData[j][1], inputData[j][1])) {
			cout << "yay"<< endl;
		}
		*/
		//outputBatch.push_back(inputData[j]);//pick random datasets from input
		if ((posOfHighestElement(inputData[j][1]) == 0) || (posOfHighestElement(inputData[j][1]) == 1) || (posOfHighestElement(inputData[j][1]) == 2) || (posOfHighestElement(inputData[j][1]) == 3) || (posOfHighestElement(inputData[j][1]) == 4) || (posOfHighestElement(inputData[j][1]) == 5) || (posOfHighestElement(inputData[j][1]) == 6) || (posOfHighestElement(inputData[j][1]) == 7) || (posOfHighestElement(inputData[j][1]) == 8) || (posOfHighestElement(inputData[j][1]) == 9)) {
			outputBatch.push_back(inputData[j]);//pick random datasets from input
		}
		else
		{
			i--;
		}
	}
}
void network::stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, double epochs, double miniBatchSize, double learningRate, const vector<vector<vector<double>>> &testData, vector<vector<vector<double>>>&weights, vector<vector<double>> &biases) {
	//example call in python >>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data) 

	for (int epoch = 0; epoch <= epochs; epoch++)
	{
		vector<vector<double>> nabla_biases(biases);
		vector<vector<vector<double>>> nabla_weights(weights);


		for (int i = 0; i < nabla_biases.size(); i++)
		{
			nabla_biases[i] = fillVecWithZeroes(nabla_biases[i]);
		}
		for (int k = 0; k < nabla_weights.size(); k++)
		{
			for (int j = 0; j< weights[k].size(); j++)
			{
				nabla_weights[k][j] = fillVecWithZeroes(nabla_weights[k][j]);
			}
		}
		vector<vector<double>> delta_nabla_biases;
		vector<vector<vector<double>>> delta_nabla_weights;
		delta_nabla_biases = nabla_biases;
		delta_nabla_weights = nabla_weights;
		vector<vector<vector<double>>> miniBatch;
		generateMiniBatch(miniBatchSize, trainingData, miniBatch);
		for (int i = 0; i < miniBatchSize; i++)
		{

			updateMiniBatch(miniBatch[i], miniBatchSize, biases, weights, learningRate, nabla_biases, nabla_weights, delta_nabla_biases, delta_nabla_weights);

		}
		double rate = (learningRate / miniBatchSize);
		//edit biases
		for (int i = 1; i < biases.size(); i++)
		{
			vector<double> newLayerBiases;
			for (int j = 0; j < biases[i].size(); j++)
			{
				//double blub;
				biases[i][j] = biases[i][j] - (rate * (delta_nabla_biases[i][j]));
				//newLayerBiases.push_back(blub);
				//biases[i][j] = blub;
			}
			//biases[i] = newLayerBiases;
		}
		//edit weights
		for (int i = 0; i < weights.size(); i++)//erstes layer ausnehmen???
		{
			vector<vector<double>> newLayerWeights;
			for (int j = 0; j < weights[i].size(); j++)
			{
				vector<double> newNodeWeights;
				for (int k = 0; k < weights[i][j].size(); k++)
				{
					//double blub;
					weights[i][j][k] = weights[i][j][k] - (rate * (delta_nabla_weights[i][j][k]));//w-(eta/length)*nablaW
																								  //newNodeWeights.push_back(blub);
																								  //weights[i][j][k] = blub;
				}
				///newLayerWeights.push_back(newNodeWeights);
			}
			//weights[i] = newLayerWeights;
		}
		//updateMiniBatch(vector<vector<double>> &dataSet, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate)
		//ladida
		//print(biases[biases.size() - 1]);
		//print(biases);
		//cout << "biases: " << endl;
		//print(biases[1]);
		//cout << sumOfVec(biases[0]);
		//cout << endl;
		//print(biases[1]);
		//cout << endl;
		cout << "Epoch " << epoch << ":";
		//cout << sumOfVec(biases[1]);
		cout << endl;
		print(biases[2]);
		cout << endl;
		cout << sumOfVec(biases[2]);
		cout << endl;
		//print(biases[3]);
		cout << endl;
		evaluate(weights, biases, trainingData);
	}
	
}
void network::feedforward(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, vector<vector<double>> &inActivations, vector<vector<double>> &weightedInputs, vector<vector<double>> &outActivations)
{	
	weightedInputs.push_back(inActivations[0]);
	outActivations.push_back(inActivations[0]);
	for (int i = 0; i < weightedInputs[0].size(); i++)
	{	
		double blub=0;
		blub = (weightedInputs[0][i]/255);
		weightedInputs[0][i] = blub;
		outActivations[0][i] = blub;
	}

	for (int i = 1; i < inBiases.size(); i++)
	{
		vector<double> weightedLayerInputs;
		vector<double> layerActivations;
		for (int j = 0; j < inBiases[i].size(); j++)
		{
			double weightedInput;
	
			weightedInput = sumOfDotProduct(outActivations[i-1], inWeights[i-1][j])+ inBiases[i-1][j];
			weightedLayerInputs.push_back(weightedInput);

		}
		weightedInputs.push_back(weightedLayerInputs);
		for (int j = 0; j < inBiases[i].size(); j++)
		{
			double nodeActivation;
			nodeActivation = sigmoid(weightedInputs[i][j]);
			layerActivations.push_back(nodeActivation);
		}



		outActivations.push_back(layerActivations);
	}

	//weighInputs
	/*
	for (int i = 0; i < inWeights.size(); i++)
	{
		vector<double> weightedLayerInputs;
		for (int j = 0; j < inWeights[i].size(); j++)
		{

			weightedLayerInputs.push_back(weightInput(weightedInputs[i], inWeights[i][j], inBiases[i + 1][j]));

		}
		weightedInputs.push_back(weightedLayerInputs);
	}*/
	//activations first layer
	/*
	for (int i = 1; i < inBiases.size(); i++)
	{
		vector<double> layerActivations;
		for (int j = 0; j < inBiases[i].size(); j++)
		{		
			
			double nodeActivation=0;
			//for (int k = 0; k < inWeights[i-1].size(); k++)
			{				
				nodeActivation = sigmoid(weightedInputs[i][j]);//sumOfDotProduct(outActivations[i-1], inWeights[i-1][j]) + inBiases[i - 1][j]) ;
				
			}
			//nodeActivation = nodeActivation + inBiases[i - 1][j];
			layerActivations.push_back(nodeActivation);
		}
		outActivations.push_back(layerActivations);
	}*/

	//outActivations.push_back(weightedInputs[0]);
	//split here in weighInputs() and activate()

	//activate
	/*
	for (int i = 0; i < weightedInputs.size(); i++)
	{
		vector<double> layerActivations;
		for (int j = 0; j < weightedInputs[i].size(); j++)
		{
			layerActivations.push_back(sigmoid(weightedInputs[i][j]));
		}
		outActivations.push_back(layerActivations);
	}*/
}
double network::weightInput(vector<double> &input, const vector<double> &weights, const double &bias) 
{

	//vector<double>outActivations;
	//double activation=0;
	double weightedInput;
	weightedInput = /*sigmoid*/(sumOfDotProduct(input, weights) + bias) ;// input.size();//wupdifuckingdoo/(input.size()
	/*for (int i = 0; i < input.size(); i++)
	{
		outActivations.push_back((input[i] * weights[i]));
	}
	for (int j = 0; j < outActivations.size(); j++)
	{
		activation = activation + outActivations[j];
	}*/
	return weightedInput;
}
void network::getDesiredOutput(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<double> &desiredOutput, vector<vector<double>> &listOfDesiredOutputs) {
	//baue element für element auf
	listOfDesiredOutputs.push_back(desiredOutput);//last layer
	for (int i = inWeights.size(); i > 0; i--)//previous layers, layer index 2-0
	{
		vector<double> previousLayerDesiredOutput;
		for (int j = 0; j < inBiases[i-1].size(); j++) // 0-30
		{
			double desiredNodeOutput = 0;
			for (int k = 0; k < inWeights[i-1].size(); k++) // 0-10
			{
				//for (int l = 0; l < inWeights[i-i][k].size(); l++) // 0-30
				{
					desiredNodeOutput = (desiredNodeOutput + ((listOfDesiredOutputs[0][k]-inBiases[i][k]) / (inWeights[i-1][k][j])));
				}
				//desiredNodeOutput = ((listOfDesiredOutputs[0][j] - inBiases[i][j]) / inWeights[i-1][j][k]);
				
			}
			previousLayerDesiredOutput.push_back(desiredNodeOutput);
		}
		listOfDesiredOutputs.insert(listOfDesiredOutputs.begin(), previousLayerDesiredOutput);
	}
	/*
	for (int i = inWeights.size(); i > 0; i--)//previous layers, layer index
	{		
		vector<double> previousLayerDesiredOutput;
		for (int j = 0; j < inWeights[i-1][0].size(); j++)//node index
		{		
			double desiredNodeOutput = 0;
			double weightSum = 0;

			for (int k = 0; k<inWeights[i-1].size();k++)//weight index
			{
				//if (inWeights[i - 1][j][k] != 0)
				//desiredNodeOutput = ((listOfDesiredOutputs[0][k] - inBiases[i][k]) / (inWeights[i - 1][k][j] / inWeights[i - 1][k].size()));
				desiredNodeOutput = desiredNodeOutput - (listOfDesiredOutputs[0][k] / inWeights[i-1][k][j]);/* (desiredOutput / weight) - bias*
				desiredNodeOutput = desiredNodeOutput + inBiases[i][k];
			}
			//for (int k = 0; k < inBiases[i].size(); k++)
			{
				//desiredNodeOutput = desiredNodeOutput - inBiases[i][j];
			}
			
		}	
		listOfDesiredOutputs.insert(listOfDesiredOutputs.begin(), previousLayerDesiredOutput);
	}*/
}
void network::getErrors(const vector<vector<double>> &realOutputs, const vector<vector<double>> &listOfDesiredOutputs, vector<vector<double>> &errors) {
	//get errors per layer, starting with last layer
	for (int i = 0; i < realOutputs.size(); i++)
	{
		vector<double>layerError;
		layerError = costDerivative(realOutputs[i] , listOfDesiredOutputs[i]);
		errors.push_back(layerError);
	}
	
		//fehlt schleife?
		
		//errors = costDerivative(/*realOutput*/, listOfDesiredOutputs[listOfDesiredOutputs.begin()]);
		//listOfDesiredOutputs.insert(listOfDesiredOutputs.begin(), errors/*listOfDesiredOutputPerLayer*/);
	
	
	//blub.push_back();
}
void network::getSigmoidPrime(const vector<vector<double>> &weightedInputs, vector<vector<double>> &sigmoidPrimeOfZ)
{
	for (int i = 0; i < weightedInputs.size(); i++)
	{
		vector<double> blub;
		for (int j = 0; j<weightedInputs[i].size() ;j++)
		{
			blub.push_back(sigmoidPrime(weightedInputs[i][j]));
		}
		sigmoidPrimeOfZ.push_back(blub);
	}
}
void network::getDelta(const vector<vector<double>> &costs, const vector<vector<double>> &sigmoidPrimeOfZ, vector<vector<double>> &delta) {
	for (int i = 0; i < costs.size(); i++)
	{	
		vector<double> deltaPerLayer;
		for (int j = 0; j < costs[i].size(); j++)
		{
			double deltaPerNode = 0;
			deltaPerNode = costs[i][j] * sigmoidPrimeOfZ[i][j];
			deltaPerLayer.push_back(deltaPerNode);
		}
		/*
		double deltaPerNode = 0;
		deltaPerNode = costs[costs.size()-2][i] * sigmoidPrimeOfZ[costs.size() - 2][i];*/
		delta.push_back(deltaPerLayer);
		//delta.insert(delta.begin(), deltaPerLayer);
		//delta.insert(delta.begin(), deltaPerLayer);
	}
}
void network::getNablaWeights(const vector<vector<double>> &delta, const vector<vector<double>> &listOfActivations, vector<vector<vector<double>>> &nabla_weights) {
	//last layer

	for (int i = 0; i < nabla_weights.size(); i++)
	{
		vector<vector<double>> bla;
		for (int j = 0; j < delta[delta.size()-(i+1)].size(); j++)
		{
			vector<double> doener;
			for (int k = 0; k < delta[delta.size()-(i+2)].size() ; k++)
			{
				double blub;
				blub = delta[delta.size() - (i + 1)][j] * listOfActivations[listOfActivations.size() - (i + 2)][k];
				doener.push_back(blub);
			}
			bla.push_back(doener);
		}
		nabla_weights[nabla_weights.size() - (i + 1)] = bla;
	}
	/*
	for (int i = 0; i < listOfActivations.size()-1; i++)
	{
		vector<vector<double>> bla;
		for (int k = 0; k < delta[delta.size() - (i + 1)].size(); k++)
		{
			vector<double> blub;
			double doener;
			
			for (int j = 0; j < listOfActivations[listOfActivations.size() - (i + 2)].size(); j++)
			{

					doener = delta[delta.size() - (i + 1)][k] * listOfActivations[listOfActivations.size() - (i + 1)][k];
					blub.push_back(doener);
				
				//doener = sumOfDotProduct(delta[delta.size() - (i + 1)], listOfActivations[listOfActivations.size() - (i + 1)]);
				//blub.push_back(doener);
			}
			bla.push_back(blub);
		}
		nabla_weights[nabla_weights.size()-(i+1)] = bla;
	}*/
	/*
	for (int i = 0; i < delta.size()-1; i++)
	{
		vector<vector<double>> layerDelta;
		for (int j = 0; j < delta[i+1].size(); j++)
		{
			vector<double> nodeDelta;
			for (int k = 0; k < listOfActivations[i].size(); k++)
			{
				nodeDelta.push_back(delta[i][k]*listOfActivations[i+1][j]);
			}
			layerDelta.push_back(nodeDelta);
		}
		nabla_weights[i] = layerDelta;
	}
	*/
	/*

	for (int k = 0; k < delta.size()-1; k++)
	{
		vector<vector<double>> deltaNode;
		for (int i = 0; i < delta[k+1].size(); i++)
		{
			vector<double> deltaWeight;
			for (int j = 0; j < delta[k].size(); j++)
			{
				deltaWeight.push_back(delta[k][j] * listOfActivations[k + 1][i]);
			}
			deltaNode.push_back(deltaWeight);
		}
		nabla_weights[k]=deltaNode;
	}*/
	/*
	for (int i = 0; i < 2; i++)
	{
		vector<vector<double>> nabla_weightLayer;
		for (int j = 0; j < listOfActivations[i+1].size(); j++)
		{
			vector<double> deltaWeight;
			for (int k = 0; k < listOfActivations[j].size(); k++)
			{				
				deltaWeight.push_back(delta[i][k]*listOfActivations[i+1][j]);

			}
			nabla_weightLayer.push_back(deltaWeight);
		}
		nabla_weights.push_back(nabla_weightLayer);
	}
	for (int i = 0; i < nabla_weights.size(); i++)
	{
		for (int j = 0; j < nabla_weights[i].size(); j++)
		{
			for (int k = 0; k < nabla_weights[i][j].size();k++)
			{
				nabla_weights[i][j][k] = delta[i][k] * listOfActivations[i][k];
			}
		}
	}
	*/
	/*
	for (int i = 0; i < delta.size()-1; i++)
	{
		//nabla_weights[i-1]
		for (int j = 0; j < nabla_weights[i].size() ; j++)
		{
			nabla_weights[i][j] = (dotProduct(delta[i], listOfActivations[i]));					
		}
	}*/

}
void network::evaluate(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<vector<vector<double>>> &inputData) {
	vector<vector<vector<double>>> testSample;
	generateMiniBatch(100, inputData, testSample);
	double numberOfSuccess=0;
	for (int i = 0; i < testSample.size(); i++)
	{	
		vector<vector<double>> result;
		vector<vector<double>> blub;
		feedforward(inWeights,inBiases,testSample[i], result, blub);
		if (hasSameHighestElement(testSample[i][1], blub[result.size()-1]))
		{
			//cout << "isSame";
			numberOfSuccess++;
		}
	}
	cout << "success rate: " << numberOfSuccess / 100<<endl;
}
int main()
{

	double epochs = 5000;
	double miniBatchSize =100;
	double learningRate =30;
	//vector<double> hyperParam{ epochs, miniBatchSize, learningRate };

	vector<vector<double>> trainingImageData;
	ReadMNISTImages(60000, 784, trainingImageData);
	vector<double> labels;
	ReadMNISTLabels(60000, labels);
	vector<int> networkSize;
	networkSize = {784, 30, 10};// getNetworkSize();
	network net{  };
	vector<vector<vector<double>>> trainingData;
	vector<vector<double>> vectorizedTrainingLabels;
	vectorizedTrainingLabels = net.vectorizeLabels(labels);
	zip(trainingImageData, vectorizedTrainingLabels, trainingData);

	net.setLayerSizes(networkSize);
	//net.print();
	vector<vector<double>> biases;
	biases = net.generateBiases(networkSize);
	//cout << "BIASES###";
	//net.print(biases);
	vector<vector<vector<double>>> weights;
	weights = net.setWeights(networkSize);
	//cout << "WEIGHTS ###";
	//net.print(weights);

	//net.print(trainingData);
	//vector<vector<vector<double>>> sample;
	//net.print(trainingData);
	net.print(biases[1]);
	cout << endl;
	//net.print(biases[2]);
	cout << endl;
	net.evaluate(weights, biases, trainingData);
	cout << "start feed"<<endl;
	net.stochasticGradientDescent(trainingData, epochs, miniBatchSize, learningRate, trainingData, weights, biases);
	/*for (int i = 0; i < 100; i++)
	{
	

	net.evaluate(weights, biases, trainingData);
	}*/
	
	//net.evalute();

	//stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, int epochs, int miniBatchSize, int learningRate, const vector<vector<vector<double>>> &testData)
	//net.print(net.vectorizeLabels(labels));
	//net.print(trainingData);
	//zip(biases[0], weights[0][0], zipped);
	//net.print(zipped);
	//net.print(allImageData);
	//net.print(labels);
	//double test = rng();
	cout << "feeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed";
	//vector<vector<double>> out;
	//out = net.feedforward(weights, biases, allImageData);
	//net.print(out);
	//net.print(zipped);
	//net.print(miniBatches);
	string blub;
	cin >> blub;
	return 0;
}