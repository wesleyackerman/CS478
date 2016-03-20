package instancebasedlearner;

import java.util.ArrayList;
import java.util.HashMap;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {
	private Matrix features;
	private Matrix labels;
	private final int kNearest = 15;
	private final double MISSING = Double.MAX_VALUE;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		this.features = features;
		this.labels = labels;
		this.features.normalize();
		this.labels.normalize();
	}
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		ArrayList<Integer> nearestNeighbors = new ArrayList<Integer>(kNearest);
		ArrayList<Double> nnDistances = new ArrayList<Double>(kNearest);
		// Set each of the NN distances to infinity, since there are no NN yet
		for (int i = 0; i < kNearest; i++) {
			nnDistances.add(i, Double.POSITIVE_INFINITY);
			nearestNeighbors.add(i, -1);
		}
		
		for (int i = 0; i < this.features.rows(); i++) {
			double distance = this.calcDistance(features, this.features.row(i));
			for (int j = 0; j < kNearest; j++) {
				if (distance < nnDistances.get(j)) {
					nnDistances.set(j, distance);
					nearestNeighbors.set(j, i);
					break;
				}
			}
		}
		double label = this.getLabel(nearestNeighbors, nnDistances);
		labels[0] = label;
	}
	
	private double calcDistance(double[] instance1, double[] instance2) {
		double totalDistance = 0;
		for (int i = 0; i < this.features.cols(); i++) {
			if (this.features.valueCount(i) == 0) {
				double dist = this.getEuclideanDistance(instance1[i], instance2[i]);
				totalDistance += Math.pow(dist, 2);
			} else {
				double dist = this.getNominalDistance(instance1[i], instance2[i]);
				totalDistance += Math.pow(dist, 2);
			}
		}
		totalDistance = Math.sqrt(totalDistance);
		return totalDistance;
	}
	
	private double getLabel(ArrayList<Integer> nearestNeighbors, ArrayList<Double> nnDistances) {
		return this.getMostCommonLabel(nearestNeighbors);
		//return this.getUnweightedRegressionLabel(nearestNeighbors, nnDistances);
		//return this.getWeightedRegressionLabel(nearestNeighbors, nnDistances);
		//return this.getWeightedMostCommonLabel(nearestNeighbors, nnDistances);
	}
	
	private double getWeightedMostCommonLabel(ArrayList<Integer> nearestNeighbors, ArrayList<Double> nnDistances) {
		HashMap<Double, Integer> labelCount = new HashMap<Double, Integer>();
		HashMap<Double, Double> weights = new HashMap<Double, Double>();
		for (int i = 0; i < this.kNearest; i++) {
			int index = nearestNeighbors.get(i);
			double label = this.labels.get(index, 0);
			int count = 1;
			double distance = nnDistances.get(i);
			
			if (labelCount.containsKey(label))
				count = labelCount.get(label);
			if (weights.containsKey(label))
				distance += weights.get(label);
			
			labelCount.put(label, (count++));
			weights.put(label, distance);
		}
		
		double bestLabel = Double.NaN;
		double bestLabelCount = 0;
		for (double label : labelCount.keySet()) {
			
			double weight = weights.get(label);
			weight /= labelCount.get(label);
			weight = (1 / Math.pow(weight, 2));
			
			if (weight * labelCount.get(label) > bestLabelCount) {
				bestLabel = label;
				bestLabelCount = weight * labelCount.get(label);
			}
		}
		return bestLabel;
	}
	
	private double getMostCommonLabel(ArrayList<Integer> nearestNeighbors) {
		HashMap<Double, Integer> labelCount = new HashMap<Double, Integer>();
		for (int i = 0; i < this.kNearest; i++) {
			int index = nearestNeighbors.get(i);
			double label = this.labels.get(index, 0);
			int count = 1;
			
			if (labelCount.containsKey(label))
				count = labelCount.get(label);
			
			labelCount.put(label, count);
		}
		
		double bestLabel = Double.NaN;
		int bestLabelCount = 0;
		for (double label : labelCount.keySet()) {
			if (labelCount.get(label) > bestLabelCount) {
				bestLabel = label;
				bestLabelCount = labelCount.get(label);
			}
		}
		return bestLabel;
	}
	
	private double getUnweightedRegressionLabel(ArrayList<Integer> nearestNeighbor, ArrayList<Double> nnDistances) {
		ArrayList<Double> weights = new ArrayList<Double>(kNearest);
		for (int i = 0; i < kNearest; i++) 
			weights.add(i, 1.0);
	
		return this.getRegressionLabel(nearestNeighbor, nnDistances, weights);
	}
	
	private double getWeightedRegressionLabel(ArrayList<Integer> nearestNeighbor, ArrayList<Double> nnDistances) {
		ArrayList<Double> weights = new ArrayList<Double>(kNearest);
		for (int i = 0; i < kNearest; i++)
			weights.add(i, (1 / Math.pow(nnDistances.get(i), 2)));
		
		return this.getRegressionLabel(nearestNeighbor, nnDistances, weights);
	}
	
	private double getRegressionLabel(ArrayList<Integer> nearestNeighbor, ArrayList<Double> nnDistances,
				ArrayList<Double> weights) {
		double weightedLabelsSum = 0;
		double weightsSum = 0;
		for (int i = 0; i < kNearest; i++) {
			int index = nearestNeighbor.get(i);
			weightedLabelsSum += (this.labels.get(index, 0) * weights.get(i));
			weightsSum += weights.get(i);
		}
		
		return (weightedLabelsSum / weightsSum);
	}
	
	// Change to a better distance metric?
		private double getNominalDistance(double instance1, double instance2) {
			if ((instance1 == MISSING) || (instance2 == MISSING))
				return .75;
			if (instance1 == instance2)
				return 0;
			else
				return 1;
		}
		
		private double getEuclideanDistance(double instance1, double instance2) {
			if ((instance1 == MISSING) || (instance2 == MISSING))
				return .75;
			return instance1 - instance2; 
		}
}
