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
	private static final double LEARNING_RATE = .1;
	private static final double MOMENTUM = 0;
	
	private int NUM_FEATURES;
	private int NUM_HIDDEN;
	private int NUM_OUTPUTS;
	
	public NeuralNet(Random rand)
	{
		this.rand = rand;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception
	{
		if (features.rows() != labels.rows())
			throw new Exception("Number of instances and number of outputs don't match");
		NUM_FEATURES = features.cols();
		final int NUM_INPUT_WEIGHTS = NUM_FEATURES + 1;
		NUM_HIDDEN = 2 * NUM_FEATURES;
		final int NUM_HIDDEN_WEIGHTS = NUM_HIDDEN + 1;
		NUM_OUTPUTS = labels.valueCount(0);
		
		inputWeights = new ArrayList<ArrayList<Double>>(NUM_INPUT_WEIGHTS);
		inputWeightsChange = new ArrayList<ArrayList<Double>>(NUM_INPUT_WEIGHTS);
		for (int i = 0; i < NUM_INPUT_WEIGHTS; i++)
		{
			inputWeights.set(i, new ArrayList<Double>(NUM_HIDDEN_WEIGHTS));
			inputWeights.set(i, new ArrayList<Double>(NUM_HIDDEN_WEIGHTS));
			for (int j = 0; j < NUM_HIDDEN_WEIGHTS; j++)
			{
				inputWeights.get(i).set(j, rand.nextGaussian()/10);
				inputWeights.get(i).set(j, (double)0);
			}
		}
		
		hiddenWeights = new ArrayList<ArrayList<Double>>(NUM_HIDDEN_WEIGHTS);
		hiddenWeightsChange = new ArrayList<ArrayList<Double>>(NUM_HIDDEN_WEIGHTS);
		for (int i = 0; i < NUM_HIDDEN_WEIGHTS; i++)
		{
			hiddenWeights.set(i, new ArrayList<Double>(NUM_OUTPUTS));
			hiddenWeightsChange.set(i, new ArrayList<Double>(NUM_OUTPUTS));
			for (int j = 0; j < NUM_OUTPUTS; i++)
			{
				hiddenWeights.get(i).set(j, rand.nextGaussian()/10);
				hiddenWeightsChange.get(i).set(j, (double)0);	
			}
		}
		
		int numInstances = features.rows();
		int numTestSet = (int)(numInstances * 0.8);
		int numValidationSet = numInstances - numTestSet;
		
		Matrix testMatrix = new Matrix(features, 0, 0, numTestSet, features.cols());
		Matrix validationMatrix = new Matrix(features, numTestSet, 0,
				features.rows()-numTestSet, features.cols());
		Matrix testLabels = new Matrix(labels, 0, 0, numTestSet, labels.cols());
		Matrix validationLabels = new Matrix(labels, numTestSet, 0,
				labels.rows()-numTestSet, labels.cols());
		
		double accuracy = 0;
		double bssf = 0;
		int epochsWithoutImprovement = 0;
		while (epochsWithoutImprovement < 5)
		{
			runEpoch(testMatrix, testLabels);
			Matrix confusion = new Matrix();
			accuracy = this.measureAccuracy(validationMatrix, validationLabels, confusion);
			if (accuracy > bssf)
			{
				bssf = accuracy;
				epochsWithoutImprovement = 0;
			}
			else
				epochsWithoutImprovement++;
		}	
	}
	
	public void runEpoch(Matrix features, Matrix labels)
	{
		int numInstances = features.rows();
		for (int i = 0; i < numInstances; i++)
			trainWithInstance(features.row(i), labels.row(i));
	}
	
	public void trainWithInstance(double[] instance, double[] labels)
	{
		ArrayList<Double> hiddenValues = new ArrayList<Double>(NUM_HIDDEN);
		ArrayList<Double> outputValues = new ArrayList<Double>(NUM_OUTPUTS);
		ArrayList<Double> outputError = new ArrayList<Double>(NUM_OUTPUTS);
		
		predictInstance(instance, hiddenValues, outputValues);
		
		ArrayList<Double> targetOutputs = new ArrayList<Double>(NUM_OUTPUTS);
		for (int i = 0; i < NUM_OUTPUTS; i++)
			targetOutputs.set(i, (double)0);
		targetOutputs.set((int)labels[0], 1.0);
		
		for (int i = 0; i < NUM_OUTPUTS; i++)
		{
			double output = outputValues.get(i);
			outputError.set(i, (targetOutputs.get(i) - output) * (output * (1 - output)));
			for (int j = 0; j <= NUM_HIDDEN; j++)
			{	
				double nodeValue = 1;
				if (j != NUM_HIDDEN)
					nodeValue = hiddenValues.get(j);
					
				double delta = (LEARNING_RATE * nodeValue * outputError.get(i))
						+ (MOMENTUM * this.hiddenWeightsChange.get(j).get(i));
				double prev = this.hiddenWeights.get(j).get(i);
				this.hiddenWeights.get(j).set(i, prev+delta);
				this.hiddenWeightsChange.get(j).set(i, delta);
			}
		}
		
		for (int i = 0; i <= NUM_HIDDEN; i++)
		{
			double output = 1;
			if (i != NUM_HIDDEN)
				output = hiddenValues.get(i);
			double error = 0;
			for (int j = 0; j < NUM_OUTPUTS; j++)
				error += outputError.get(j) * this.hiddenWeights.get(i).get(j);
			error *= (output * (1 - output));
			
			for (int j = 0; j <= NUM_FEATURES; j++)
			{
				double nodeValue = 1;
				if (j != NUM_FEATURES)
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
		for (int i = 0; i <= NUM_HIDDEN; i++)
		{
			double sum = 0;
			for (int j = 0; j <= NUM_FEATURES; j++)
			{
				if (j == NUM_FEATURES)
					sum += 1 * inputWeights.get(j).get(i);
				else
					sum += instance[j] * inputWeights.get(j).get(i);
			}
			double output = 1 / (1 + Math.pow(Math.E, (0 - sum)));
			hiddenValues.set(i, output);
		}
		
		for (int i = 0; i < NUM_OUTPUTS; i++)
		{
			double sum = 0;
			for (int j = 0; j <= NUM_HIDDEN; j++)
			{
				if (j == NUM_HIDDEN)
					sum += 1 * hiddenWeights.get(j).get(i);
				else
					sum += hiddenValues.get(j) * hiddenWeights.get(j).get(i);
			}
			double output = 1 / (1 + Math.pow(Math.E, (0 - sum)));
			outputValues.set(i, output);
		}
	}
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception
	{
		ArrayList<Double> hiddenValues = new ArrayList<Double>(NUM_HIDDEN);
		ArrayList<Double> outputValues = new ArrayList<Double>(NUM_OUTPUTS);
		
		predictInstance(features, hiddenValues, outputValues);
		for (int i = 0; i < NUM_OUTPUTS; i++)
			labels[i] = outputValues.get(i);
	}	
}
