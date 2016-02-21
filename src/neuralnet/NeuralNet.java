package neuralnet;

import java.util.ArrayList;
import java.util.Random;

import toolkit.*;

public class NeuralNet extends SupervisedLearner
{
	private Random rand;
	private ArrayList<ArrayList<Double>> inputWeights;
	private ArrayList<ArrayList<Double>> hiddenWeights;
	private ArrayList<ArrayList<Double>> inputWeightsChange;
	private ArrayList<ArrayList<Double>> hiddenWeightsChange;
	
	private static final double LEARNING_RATE = .05;
	private static final double MOMENTUM = .5;
	private static final int WINDOW_WITHOUT_IMPROVEMENT = 30;
	
	private int numFeatures;
	private int numHidden;
	private int numOutputs;
	private int numInputWeights;
	private int numHiddenWeights;
	
	public NeuralNet(Random rand)
	{
		this.rand = rand;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception
	{
		if (features.rows() != labels.rows())
			throw new Exception("Number of instances and number of outputs don't match");
		numFeatures = features.cols();
		numInputWeights = numFeatures + 1;
		//numHidden = 2 * numFeatures;
		numHidden = 40;
		numHiddenWeights = numHidden + 1;
		numOutputs = labels.valueCount(0);
		
		this.initWeights();
		features.shuffle(rand, labels);
				
		int numInstances = features.rows();
		int numTestSet = (int)(numInstances * 0.8);
		int numValidationSet = numInstances - numTestSet;
		
		Matrix trainingMatrix = new Matrix(features, 0, 0, numTestSet, features.cols());
		Matrix validationMatrix = new Matrix(features, numTestSet, 0,
				features.rows()-numTestSet, features.cols());
		Matrix trainingLabels = new Matrix(labels, 0, 0, numTestSet, labels.cols());
		Matrix validationLabels = new Matrix(labels, numTestSet, 0,
				numValidationSet, labels.cols());
		
		double accuracy = 0;
		double bssf = 0;
		int epochsWithoutImprovement = 0;
		int epochsRun = 0;
		while (epochsWithoutImprovement < WINDOW_WITHOUT_IMPROVEMENT)
		{
			epochsRun++;
			runEpoch(trainingMatrix, trainingLabels);
			Matrix confusion = new Matrix();
			accuracy = this.measureAccuracy(validationMatrix, validationLabels, confusion);
			if (accuracy > bssf)
			{
				bssf = accuracy;
				epochsWithoutImprovement = 0;
			}
			else
				epochsWithoutImprovement++;
			//System.out.println("Accuracy: " + accuracy);
			trainingMatrix.shuffle(rand, trainingLabels);
		}

		//Calc MSE
		double trainSetMSE = this.getMSE(numTestSet, trainingMatrix, trainingLabels);
		double validationSetMSE = this.getMSE(numValidationSet, validationMatrix, validationLabels);
		Matrix confusion = new Matrix();
		double validationSetAccuracy = this.measureAccuracy(validationMatrix, validationLabels, confusion);
		
		System.out.println("# Epochs to get best validation acc: " + (epochsRun - WINDOW_WITHOUT_IMPROVEMENT));
		System.out.println("Training Set MSE: " + trainSetMSE);
		System.out.println("Validation Set MSE: " + validationSetMSE);
		System.out.println("Validation Set Accuracy: " + validationSetAccuracy);
	}
	
	public void runEpoch(Matrix features, Matrix labels)
	{
		int numInstances = features.rows();
		for (int i = 0; i < numInstances; i++)
			trainWithInstance(features.row(i), labels.row(i));
	}
	
	public void trainWithInstance(double[] instance, double[] labels)
	{
		ArrayList<Double> hiddenValues = new ArrayList<Double>(numHidden);
		ArrayList<Double> outputValues = new ArrayList<Double>(numOutputs);
		ArrayList<Double> outputError = new ArrayList<Double>(numOutputs);
		
		predictInstance(instance, hiddenValues, outputValues);
		
		ArrayList<Double> targetOutputs = new ArrayList<Double>(numOutputs);
		for (int i = 0; i < numOutputs; i++)
			targetOutputs.add((double)0);
		targetOutputs.set((int)labels[0], 1.0);
		
		// Backpropagate the error. Place this in separate function?
		for (int i = 0; i < numOutputs; i++)
		{
			double output = outputValues.get(i);
			outputError.add((targetOutputs.get(i) - output) * (output * (1 - output)));
			for (int j = 0; j <= numHidden; j++)
			{	
				double nodeValue = 1;
				if (j != numHidden)
					nodeValue = hiddenValues.get(j);
					
				double delta = (LEARNING_RATE * nodeValue * outputError.get(i))
						+ (MOMENTUM * this.hiddenWeightsChange.get(j).get(i));
				double prev = this.hiddenWeights.get(j).get(i);
				this.hiddenWeights.get(j).set(i, prev+delta);
				this.hiddenWeightsChange.get(j).set(i, delta);
			}
		}
		
		for (int i = 0; i <= numHidden; i++)
		{
			double output = 1;
			if (i != numHidden)
				output = hiddenValues.get(i);
			double error = 0;
			for (int j = 0; j < numOutputs; j++)
				error += outputError.get(j) * this.hiddenWeights.get(i).get(j);
			error *= (output * (1 - output));
			
			for (int j = 0; j <= numFeatures; j++)
			{
				double nodeValue = 1;
				if (j != numFeatures)
					nodeValue = instance[j];
				double delta = (LEARNING_RATE * nodeValue * error) +
						(MOMENTUM * this.inputWeightsChange.get(j).get(i));
				double prev = this.inputWeights.get(j).get(i);
				this.inputWeights.get(j).set(i, prev+delta);
				this.inputWeightsChange.get(j).set(i, delta);			
			}
		}
	}
	
	private void predictInstance(double[] instance, ArrayList<Double> hiddenValues,
			ArrayList<Double> outputValues)
	{
		for (int i = 0; i < numHidden; i++)
		{
			double sum = 0;
			for (int j = 0; j <= numFeatures; j++)
			{
				if (j == numFeatures)
					sum += 1 * inputWeights.get(j).get(i);
				else
					sum += instance[j] * inputWeights.get(j).get(i);
			}
			double output = 1 / (1 + Math.pow(Math.E, (0 - sum)));
			hiddenValues.add(output);
		}
		
		for (int i = 0; i < numOutputs; i++)
		{
			double sum = 0;
			for (int j = 0; j <= numHidden; j++)
			{
				if (j == numHidden)
					sum += 1 * hiddenWeights.get(j).get(i);
				else
					sum += hiddenValues.get(j) * hiddenWeights.get(j).get(i);
			}
			double output = 1 / (1 + Math.pow(Math.E, (0 - sum)));
			outputValues.add(i, output);
		}
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception
	{
		ArrayList<Double> hiddenValues = new ArrayList<Double>(numHidden);
		ArrayList<Double> outputValues = new ArrayList<Double>(numOutputs);
		
		predictInstance(features, hiddenValues, outputValues);

		double largestOutput = 0;
		int largestOutputIndex = -1;
		for (int i = 0; i < numOutputs; i++)
		{
			//System.out.print(outputValues.get(i) + " ");
			if (outputValues.get(i) > largestOutput)
			{
				largestOutput = outputValues.get(i);
				largestOutputIndex = i;
			}
		}
		//System.out.println("Largest Index: " + largestOutputIndex);
		labels[0] = largestOutputIndex;
	}
		
	private void initWeights()
	{
		inputWeights = new ArrayList<ArrayList<Double>>(numInputWeights);
		inputWeightsChange = new ArrayList<ArrayList<Double>>(numInputWeights);
		for (int i = 0; i < numInputWeights; i++)
		{
			inputWeights.add(new ArrayList<Double>(numHiddenWeights));
			inputWeightsChange.add(new ArrayList<Double>(numHiddenWeights));
			for (int j = 0; j < numHiddenWeights; j++)
			{
				inputWeights.get(i).add(rand.nextGaussian());//% (1/(Math.sqrt((int)numFeatures))));
				inputWeightsChange.get(i).add((double)0);
			}
		}
		
		hiddenWeights = new ArrayList<ArrayList<Double>>(numHiddenWeights);
		hiddenWeightsChange = new ArrayList<ArrayList<Double>>(numHiddenWeights);
		for (int i = 0; i < numHiddenWeights; i++)
		{
			hiddenWeights.add(new ArrayList<Double>(numOutputs));
			hiddenWeightsChange.add(new ArrayList<Double>(numOutputs));
			for (int j = 0; j < numOutputs; j++)
			{
				hiddenWeights.get(i).add(rand.nextGaussian());// % (1/(Math.sqrt((int)numFeatures))));
				hiddenWeightsChange.get(i).add((double)0);	
			}
		}		
	}
	
	public double getMSE(int numInSet, Matrix set, Matrix labels)
	{
		double sum = 0;
		for (int i = 0; i < numInSet; i++)
		{
			ArrayList<Double> hiddenValues = new ArrayList<Double>(numHidden);
			ArrayList<Double> outputValues = new ArrayList<Double>(numOutputs);
			this.predictInstance(set.row(i), hiddenValues, outputValues);
			
			ArrayList<Double> targetOutputs = new ArrayList<Double>(numOutputs);
			for (int j = 0; j < numOutputs; j++)
				targetOutputs.add((double)0);
			targetOutputs.set((int)labels.row(i)[0], 1.0);
			
			double distance = 0;
			// Pythagorean theorem in 3 dimensions: distance = sqrt(x^2 + y^2 + z^2)
			for (int j = 0; j < numOutputs; j++)
				distance += Math.pow((targetOutputs.get(j) - outputValues.get(j)), 2);
			//Would take sqrt to get true distance, but for MSE we square it again anyway.
			sum += distance;
		}
		sum /= (numInSet - 2);
		return sum;
	}
}
