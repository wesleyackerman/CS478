package perceptron;

import java.util.ArrayList;
import java.util.Random;

import toolkit.*;

public class Perceptron extends SupervisedLearner {
	private Random rand;
	private ArrayList<Double> weights;
	private static final double LEARNING_RATE = .1;
	private static final double THRESHOLD = 0;
	private static final int EPOCHS_WITHOUT_IMPROVEMENT = 10;

	public Perceptron(Random rand) {
		this.rand = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception
	{
		if (features.rows() != labels.rows())
			throw new Exception("Number of instances and number of outputs don't match");
		int numFeatures = features.cols();
		int numWeights = numFeatures + 1;
		
		weights = new ArrayList<Double>(numWeights);
		initWeights(numWeights);
		
		double bestPercentageCorrect = 0;
		int epochsWithoutImprovement = 0;
		int numEpochsRun = 0;
		
		while (epochsWithoutImprovement < EPOCHS_WITHOUT_IMPROVEMENT)
		{
			numEpochsRun++;
			double percentageCorrect = runEpoch(features, labels);
			System.out.println((1 - percentageCorrect));
			if (percentageCorrect > bestPercentageCorrect)
			{
				bestPercentageCorrect = percentageCorrect;
				epochsWithoutImprovement = 0;
			}
			else
				epochsWithoutImprovement++;
		}
		for (int i = 0; i < weights.size(); i++)
			System.out.println("Weight " + i + ": " + weights.get(i));
		System.out.println("Number of epochs run: " + numEpochsRun);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception
	{
		double output = predictInstance(features);
		labels[0] = output;	
	}
	
	private double runEpoch(Matrix features, Matrix labels) throws Exception
	{
		int numInstances = features.rows();
		int correctCount = 0;

		for (int i = 0; i < numInstances; i++)
		{
			boolean predictionCorrect = trainWithInstance(features.row(i), labels.get(i, 0));
			if (predictionCorrect)
				correctCount++;
		}
		double percentageCorrect = (double)correctCount / (double)features.rows();
		
		return percentageCorrect;
	}
	
	/*
	 * Returns whether or not prediction of instance's output by perceptron was correct.
	 * If incorrect, calls updateWeights to adjust weights accordingly
	 */
	private boolean trainWithInstance(double[] instance, double targetOutput) throws Exception
	{
		if ((targetOutput != 1) && (targetOutput != 0))
			throw new Exception("Labels must be a 0 or 1");
		
		double output = predictInstance(instance);
		boolean predictionCorrect = true;
		
		if (output == targetOutput)
			return predictionCorrect;
		
		updateWeights(instance, output, targetOutput);
		return (predictionCorrect = false);
	}
	
	private double predictInstance(double[] instance)
	{
		int index = 0;
		double result = 0;
		
		for (double attribute : instance)
		{
			result += attribute * weights.get(index);
			index++;
		}
		// Bias weight, multiplied by 1
		result += weights.get(index);
		
		double output = 0;
		if (result > THRESHOLD)
			output = 1;
		
		return output;
	}

	private void initWeights(int numWeights)
	{
		for (int i = 0; i < numWeights; i++)
			weights.add(i, rand.nextDouble());
	}
	
	private void updateWeights(double[] instance, double output, double targetOutput)
	{
		for (int i = 0; i < weights.size(); i++)
		{
			double featureValue = 1;
			if (i < (weights.size() - 1))
				featureValue = instance[i];
			weights.set(i, weights.get(i) + ((targetOutput - output) * featureValue * LEARNING_RATE));
		}
	}
	//change in weight = (target - output)*learningRate*attribute
}
