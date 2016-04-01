package clustering;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class KMeansCluster extends SupervisedLearner {
	private final static double MISSING = Double.MAX_VALUE;
	private int numClusters;
	private ArrayList<ArrayList<Double>> centroids;
	private ArrayList<ArrayList<Integer>> clusterLists;
	private Matrix features;
	private Matrix labels;
	
	public KMeansCluster(int k) {
		numClusters = k;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		centroids = new ArrayList<ArrayList<Double>>(numClusters);
		clusterLists = new ArrayList<ArrayList<Integer>>(numClusters);
		for (int i = 0; i < numClusters; i++) {
			centroids.add(i, new ArrayList<Double>(features.cols()));
		}
		for (int i = 0; i < numClusters; i++) {
			clusterLists.add(i, new ArrayList<Integer>());
		}
		
		ArrayList<Integer> usedRows = new ArrayList<Integer>();
		Random rand = new Random();
		while(usedRows.size() < numClusters) {
			int row = rand.nextInt() % features.rows();
			if (!usedRows.contains(row)) {
				usedRows.add(row);
				for (int i = 0; i < features.cols(); i++) {
					centroids.get(usedRows.size()).add(i, features.get(row, i));
				}
			}
		}
		boolean centroidHasChanged = true;
		
		while(centroidHasChanged) { //while centroids keep moving
			for (int i = 0; i < features.rows(); i++) {
				
				double[] instance = features.row(i);
				int closestCentroid = -1;
				double closestDistance = Double.POSITIVE_INFINITY;
				
				for (int j = 0; j < numClusters; j++) {
					ArrayList<Double> centroid = this.centroids.get(j);
					Double[] centroidArray = new Double[centroid.size()];
					double distance = this.getDistanceBetweenInstances(centroid.toArray(centroidArray), instance);
					
					if (distance < closestDistance) {
						closestDistance = distance;
						closestCentroid = j;
					}
				}
				this.clusterLists.get(closestCentroid).add(i);
			}
			
			ArrayList<ArrayList<Double>> oldCentroids = new ArrayList<ArrayList<Double>>(this.centroids);
			for (int i = 0; i < numClusters; i++) {
				ArrayList<Integer> clusterInstances = this.clusterLists.get(i);
				for (int j = 0; j < features.cols(); j++) {
					double newCentroidValue = this.getColumnAverage(j, clusterInstances);
					this.centroids.get(i).set(j, newCentroidValue);
				}
			}
			centroidHasChanged = this.centroidHasChanged(oldCentroids);
		}	
	}
	
	private boolean centroidHasChanged(ArrayList<ArrayList<Double>> oldCentroids) {
		for (int i = 0; i < numClusters; i++) {
			ArrayList<Double> oldCentroid = oldCentroids.get(i);
			ArrayList<Double> newCentroid = this.centroids.get(i);
			Double[] oldArray = new Double[oldCentroid.size()];
			Double[] newArray = new Double[newCentroid.size()];
			double distance =
					this.getDistanceBetweenInstances(oldCentroid.toArray(oldArray), newCentroid.toArray(newArray));
			if (distance != 0) {
				return true;
			}
		}
		return false;
	}
	
	private double getColumnAverage(int column, ArrayList<Integer> instances) {
		int valueCount = this.features.valueCount(column);
		if (valueCount == 0) {
			double sum = 0;
			int count = 0;
			for (int i = 0; i < instances.size(); i++) {
				int index = instances.get(i);
				if (this.features.get(index, column) != MISSING) {
					sum += this.features.get(index, column);
					count++;
				}
			}
			return (sum / count);
		} else {
			ArrayList<Integer> attributeValueCount = new ArrayList<Integer>(valueCount);
			for (int i = 0; i < valueCount; i++) {
				attributeValueCount.add(0);
			}
			for (int i = 0; i < instances.size(); i++) {
				int index = instances.get(i);
				double value = this.features.get(index, column);
				if (value != MISSING) {
					int count = attributeValueCount.get((int)value);
					attributeValueCount.set((int)value, (count+1));
				}
			}
			
			int bestIndex = -1;
			int mostInstances = 0;
			for (int i = 0; i < valueCount; i++) {
				int count = attributeValueCount.get(i);
				if (count > mostInstances) {
					mostInstances = count;
					bestIndex = i;
				}
			}
			return bestIndex;
		}
	}
	
	private double getDistanceBetweenInstances(Double[] centroid, double[] instance) {
		double totalDistance = 0;
		for(int i = 0; i < this.features.cols(); i++) {
			double distance = getDistance(centroid[i], instance[i], features.valueCount(i));
			totalDistance += Math.pow(distance, 2);
		}
		totalDistance = Math.sqrt(totalDistance);
		return totalDistance;
	}
	
	private double getDistance(double x1, double x2, int valueCount) {
		if ((x1 == MISSING) || (x2 == MISSING)) {
			return 1;
		} else if (valueCount == 0) {
			return x1-x2;
		} else {
			if (x1 == x2)
				return 0;
			else
				return 1;
		}
	}
	
	private double getDistanceBetweenInstances(Double[] centroid, Double[] instance) {
		double totalDistance = 0;
		for(int i = 0; i < this.features.cols(); i++) {
			double distance = getDistance(centroid[i], instance[i], features.valueCount(i));
			totalDistance += Math.pow(distance, 2);
		}
		totalDistance = Math.sqrt(totalDistance);
		return totalDistance;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {}

}
