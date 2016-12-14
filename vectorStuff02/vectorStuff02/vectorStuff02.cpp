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
	void backdrop(const int &miniBatchSize,const int &learningRate, vector<double> &activationsFromPreviousLayer, vector<double> &desiredOutput, vector<vector<vector<double>>> &inWeights, vector<vector<double>> &inBiases, vector<vector<double>> &nabla_biases, vector<vector<vector<double>>> &nabla_weights);
	vector<vector<double>> vectorizeLabels(const vector<double> &labels);
	void updateMiniBatch(const vector<vector<double>> &miniBatch, const int &miniBatchSize, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRat);
	//void generateMiniBatch(int miniBatchSize, const vector<vector<vector<double>>> &inputData, vector<vector<vector<double>>> &outputBatch);
	void stochasticGradientDescent(const vector<vector<vector<double>>> &trainingData, int epochs, int miniBatchSize, int learningRate, const vector<vector<vector<double>>> &testData, vector<vector<vector<double>>>&weights, vector<vector<double>> &biases);
	double getActivation(vector<double> &input, const vector<double> &weights);
	void getDesiredOutput(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<double> &desiredOutput, vector<vector<double>> &listOfDesiredOutputs);
	void getErrors(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<vector<double>> &realOutputs, const vector<vector<double>> &listOfDesiredOutputs, vector<vector<double>> &errors);
	void getSigmoidPrime(const vector<vector<double>> &weightedInputs, vector<vector<double>> &sigmoidPrimeofZ);
	void getDelta(const vector<vector<double>> &costs, const vector<vector<double>> &sigmoidPrimeOfZ, vector<double> &delta);
	void getNablaWeights(const vector<double> &delta, const vector<vector<double>> &listOfActivations, vector<vector<vector<double>>> &nabla_weights);
	

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
	return (1.0 / (1.0 + (exp(-z))));
	/*
	double x;
	x = (1 + (exp((-1) * z)));
	return (1 / x);*/
}
double sigmoidPrime(double z) {
	return (sigmoid(z) * (1.0 - sigmoid(z)));
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
	vector<vector<double>> delta_nabla_biases;
	vector<vector<vector<double>>> delta_nabla_weights;
	vector<vector<double>> new_biases;
	vector<vector<vector<double>>> new_weights;
	vector<double> delta;
	
	//vector<vector<double>> blub;
	listOfActivations.push_back(activationsFromPreviousLayer);
	
	//feedforward()

	feedforward(inWeights, inBiases, listOfActivations, weightedInputs, calculatedActivations); //calculates activations
	for (int i = 1; i < calculatedActivations.size(); i++) {
		listOfActivations.push_back(calculatedActivations[i]);
	}
	//***backward pass***
	getDesiredOutput(inWeights, inBiases, desiredOutput, listOfDesiredOutputs);
	//get costs
	getErrors(inWeights, inBiases,listOfActivations, listOfDesiredOutputs, costs);// compare activations to desiredOutputs
	//secondToLast = myVector[myVector.size() - 2];

	//get SigmoidPrime of Zs via Zs from last Layer to calculate derivative of costs ()
	//for each (vector<double> layerActivation in calculatedActivations)
	getSigmoidPrime(weightedInputs, sigmoidPrimeOfZ); //HIER GEHTS WEITER
	
	//delta= costs[vorletztes layer] * sigmoidPrimeOfZ[vorletztes layer]

	getDelta(costs, sigmoidPrimeOfZ, delta);

	nabla_biases[nabla_biases.size() - 2] = delta;
	getNablaWeights(delta, listOfActivations, nabla_weights);
	//nabla_weights[nabla_weights.size() - 2] = dotProduct(delta, listOfActivations[listOfActivations.size()-2]);


	delta_nabla_biases = nabla_biases;
	delta_nabla_weights = nabla_weights;
	new_biases = nabla_biases;
	new_weights = nabla_weights;

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
	}
	inWeights = new_weights;
	inBiases = new_biases;


}
void network::updateMiniBatch(const vector<vector<double>> &dataSet, const int &miniBatchSize, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate)
{
	
	vector<vector<double>> nabla_biases(biases);
	vector<vector<vector<double>>> nabla_weights(weights);

	for (int i = 0; i < nabla_biases.size(); i++)
	{
		nabla_biases[i] = fillVecWithZeroes(nabla_biases[i]);
	}
	for (int k = 0; k < nabla_weights.size(); k++)
	{
		for (int j = 0;  j< weights[k].size(); j++)
		{
			nabla_weights[k][j]=fillVecWithZeroes(nabla_weights[k][j]);
		}
	}

	for (int i = 0; i < miniBatchSize; i++) //muss noch minibatch_size werden
	{
		//vectors delta__nabla_weights, delta_nabla_biases = output of backdrop
		//vector<double> x;
		vector<double>activationsAtInputLayer(dataSet[0]);
		vector<double>output(dataSet[1]);
		network::backdrop(miniBatchSize,learningRate, activationsAtInputLayer, output, weights, biases, nabla_biases, nabla_weights);
		
	}
	//biases = (bias-(learningRate/))

	
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
			
			
			updateMiniBatch(dataSet, miniBatchSize, biases,weights,learningRate);
			//updateMiniBatch(vector<vector<double>> &dataSet, vector<vector<double>> &biases, vector<vector<vector<double>>> &weights, int learningRate)
			//ladida
		}
	}

}
void network::feedforward(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, vector<vector<double>> &inActivations, vector<vector<double>> &weightedInputs, vector<vector<double>> &outActivations)
{
	
	//weightedInputs.push_back(inActivations[0]);
	outActivations.push_back(inActivations[0]);
	//split here in weighInputs() and activate()

	//weighInputs
	for (int i = 0; i < inWeights.size(); i++)
	{
		vector<double> weightedLayerInputs;
		for (int j = 0; j < inWeights[i].size(); j++)
		{
			/*for (int k = 0; k<inWeights[i][j][k]; k++)
			{
				double blub;
				blub += inWeights[i][j][k];
			}*/
			double z;
			z = sumOfDotProduct(inWeights[i][j], outActivations[i]); //0 is falsch
			weightedLayerInputs.push_back(z);
		}
		weightedInputs.push_back(weightedLayerInputs);
	
	//activate
		vector<double> layerActivations;
		for (int j = 0; j < inWeights[i].size(); j++)
		{
			layerActivations.push_back(getActivation(outActivations[i], inWeights[i][j])+ inBiases[i][j]);
		}
		outActivations.push_back(layerActivations);
	}
}
double network::getActivation(vector<double> &input, const vector<double> &weights) 
{
	vector<double>outActivations;
	double activation=0;
	for (int i = 0; i < input.size(); i++)
	{
		outActivations.push_back((input[i] * weights[i]));
	}
	for (int j = 0; j < outActivations.size(); j++)
	{
		activation = activation + outActivations[j];
	}
	return activation;
}
void network::getDesiredOutput(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases, const vector<double> &desiredOutput, vector<vector<double>> &listOfDesiredOutputs) {
	//baue element für element auf
	listOfDesiredOutputs.push_back(desiredOutput);//last layer

	for (int i = inWeights.size(); i > 1; i--)//previous layers, layer index
	{		
		vector<double> previousLayerDesiredOutput;
		for (int j=0; j<inWeights[i-1][0].size(); j++)//node index
		{		
			double desiredNodeOutput = 0;
			for (int k=0; k<inWeights[i-1].size();k++)//weight index
			{
				//if (inWeights[i - 1][j][k] != 0)
					
				desiredNodeOutput = desiredNodeOutput+(listOfDesiredOutputs[0][k] / inWeights[i-1][k][j]);/* (desiredOutput / weight) - bias*/
			}
			desiredNodeOutput = desiredNodeOutput - inBiases[i - 1][j];
			previousLayerDesiredOutput.push_back(desiredNodeOutput);
		}	
		listOfDesiredOutputs.insert(listOfDesiredOutputs.begin(), previousLayerDesiredOutput);
	}
}
void network::getErrors(const vector<vector<vector<double>>> &inWeights, const vector<vector<double>> &inBiases,const vector<vector<double>> &realOutputs, const vector<vector<double>> &listOfDesiredOutputs, vector<vector<double>> &errors) {
	//get errors per layer, starting with last layer
	for (int i = 1; i < realOutputs.size(); i++)
	{
		vector<double>layerError;
		layerError = costDerivative(realOutputs[i] , listOfDesiredOutputs[i-1]);
		errors.push_back(layerError);
	}
	
		//fehlt schleife?
		
		//errors = costDerivative(/*realOutput*/, listOfDesiredOutputs[listOfDesiredOutputs.begin()]);
		//listOfDesiredOutputs.insert(listOfDesiredOutputs.begin(), errors/*listOfDesiredOutputPerLayer*/);
	
	
	//blub.push_back();
}
void network::getSigmoidPrime(const vector<vector<double>> &weightedInputs, vector<vector<double>> &sigmoidPrimeOfZ){
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
void network::getDelta(const vector<vector<double>> &costs, const vector<vector<double>> &sigmoidPrimeOfZ, vector<double> &delta) {
	for (int i = 0; i < costs[costs.size()-2].size(); i++)
	{
		double deltaPerNode = 0;
		deltaPerNode = costs[costs.size()-2][i] * sigmoidPrimeOfZ[costs.size() - 2][i];
		delta.push_back(deltaPerNode);
	}

}
void network::getNablaWeights(const vector<double> &delta, const vector<vector<double>> &listOfActivations, vector<vector<vector<double>>> &nabla_weights) {
	for (int i = 1; i < listOfActivations.size()-2; i++)
	{
		//nabla_weights[i-1]
		for (int j = 0; j < nabla_weights[i-1].size() ; j++)
		{
			nabla_weights[i - 1][j]=(dotProduct(delta, listOfActivations[listOfActivations.size() - 2]));
		}
	}

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