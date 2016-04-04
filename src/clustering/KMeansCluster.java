package clustering;

import java.util.ArrayList;
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
		this.features = features;
		this.labels = labels;
		
		centroids = new ArrayList<ArrayList<Double>>(numClusters);
		clusterLists = new ArrayList<ArrayList<Integer>>(numClusters);
		for (int i = 0; i < numClusters; i++) {
			centroids.add(i, new ArrayList<Double>(features.cols()));
		}
		for (int i = 0; i < numClusters; i++) {
			clusterLists.add(i, new ArrayList<Integer>());
		}
		
		/*for (int i = 0; i < this.numClusters; i++) {
			System.out.println("\nCentroid " + i + ": ");
			ArrayList<Double> centroid = this.centroids.get(i);
			for (int j = 0; j < features.cols(); j++) {
				centroid.add(features.get(i, j));
				
				if (features.get(i, j) == MISSING)
					System.out.print("?" + " ");
				else
					System.out.print(features.get(i, j) + " ");
			}
		}*/
		this.centroids = this.getRandomCentroids();
		System.out.println();
				
		boolean centroidHasChanged = true;	
		while(centroidHasChanged) {
			System.out.println("************************************");
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
				System.out.print(i + ":" + closestCentroid + ",");
				this.clusterLists.get(closestCentroid).add(i);
			}
			
			ArrayList<ArrayList<Double>> oldCentroids = this.copyCentroidArray();
			ArrayList<ArrayList<Double>> newCentroids = this.copyCentroidArray(); // Array to hold averages (centroid) for medoid calculation
			for (int i = 0; i < numClusters; i++) {
				ArrayList<Integer> clusterInstances = this.clusterLists.get(i);
				for (int j = 0; j < features.cols(); j++) {
					double newCentroidValue = this.getColumnAverage(j, clusterInstances);
					newCentroids.get(i).set(j, newCentroidValue);
				}
			}
			
			//------------------------Addition for medoids---------------------------
			for (int i = 0; i < this.numClusters; i++) {
				double smallestDist = Double.POSITIVE_INFINITY;
				int newMedoidIndex = -1;
				ArrayList<Double> centroid = newCentroids.get(i);
				for (int j = 0; j < this.features.rows(); j++) {
					double[] instance = this.features.row(j);
					Double[] centroidArray = new Double[centroid.size()];
					double distance = this.getDistanceBetweenInstances(centroid.toArray(centroidArray), instance);
					if (distance < smallestDist) {
						smallestDist = distance;
						newMedoidIndex = j;
					}
				}
				
				double[] newMedoid = this.features.row(newMedoidIndex);
				for (int j = 0; j < this.features.cols(); j++) {
					this.centroids.get(i).set(j, newMedoid[j]);
				}
			}
			//-------------------------------------------------------------------
			
			this.printCentroids();
			centroidHasChanged = this.centroidHasChanged(oldCentroids);
			
			if (centroidHasChanged) {
				clusterLists = new ArrayList<ArrayList<Integer>>(numClusters);
				for (int i = 0; i < numClusters; i++) {
					clusterLists.add(i, new ArrayList<Integer>());
				}
			}
		}
		double silhouette = this.calcTotalSilhouette();
		this.printResults(silhouette);
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
			if (count != 0)
				return (sum / count);
			return 0;
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
	
	private double calcClusterSSE(int cluster) {
		double SSE = 0;
		ArrayList<Double> centroid = this.centroids.get(cluster);
		Double[] centroidArray = new Double[centroid.size()];
		
		ArrayList<Integer> clusterList = this.clusterLists.get(cluster);
		for (int i = 0; i < clusterList.size(); i++) {
			int index = clusterList.get(i);
			double[] instance = this.features.row(index);
			SSE += Math.pow(this.getDistanceBetweenInstances(centroid.toArray(centroidArray), instance), 2);
		}
		return SSE;
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
	
	private void printResults(double silhouette) {
		double totalSSE = 0;
		System.out.println("Number of clusters: " + this.numClusters);
		for (int i = 0; i < this.numClusters; i++) {
			ArrayList<Double> centroid = this.centroids.get(i);
			System.out.print("Centroid " + i + ": ");
			for (int j = 0; j < centroid.size(); j++) {
				if (centroid.get(j) == MISSING)
					System.out.print("?" + " ");
				else if (features.valueCount(j) == 0)
					System.out.print(centroid.get(j) + " ");
				else
					System.out.print(features.attrValue(j, (int)(double)centroid.get(j)) + " ");
			}
			System.out.println("\nNumber of instances: " + this.clusterLists.get(i).size());
			double clusterSSE = this.calcClusterSSE(i);
			totalSSE += clusterSSE;
			System.out.println("Cluster SSE: " + clusterSSE);
		}
		System.out.println("Total SSE: " + totalSSE);
		System.out.println("Silhouette: " + silhouette);
	}
	
	private ArrayList<ArrayList<Double>> copyCentroidArray() {
		ArrayList<ArrayList<Double>> copy = new ArrayList<ArrayList<Double>>(numClusters);
		for (int i = 0; i < numClusters; i++) {
			copy.add(i, new ArrayList<Double>(features.cols()));
		}
		
		for (int i = 0; i < this.numClusters; i++) {
			ArrayList<Double> copyCentroid = copy.get(i);
			ArrayList<Double> centroid = this.centroids.get(i);
			for (int j = 0; j < features.cols(); j++) {
				copyCentroid.add(centroid.get(j));
			}
		}
		return copy;
	}
	
	private double calcTotalSilhouette() {
		double totalSilhouette = 0;
		for (int i = 0; i < this.numClusters; i++) {
			totalSilhouette += this.calcSilhouetteCluster(i);
		}
		return (totalSilhouette / this.numClusters);
	}
	
	private double calcSilhouetteCluster(int cluster) {
		ArrayList<Integer> clusterList = this.clusterLists.get(cluster);
		double totalSilhouette = 0;
		for (int i = 0; i < clusterList.size(); i++) {
			double[] instance = this.features.row(clusterList.get(i));
			totalSilhouette += this.calcSilhouetteInstance(instance, cluster);
		}
		return (totalSilhouette / clusterList.size());
	}
	
	private double calcSilhouetteInstance(double[] instance, int cluster) {
		double a = this.silhouetteCalcA(instance, cluster);
		double b = this.silhouetteCalcB(instance, cluster);
		double silhouette = (b - a) / Math.max(a, b);
		return silhouette;
	}
	
	private double silhouetteCalcA(double[] instance, int cluster) {
		ArrayList<Integer> instances = this.clusterLists.get(cluster);
		double distance = 0;
		for (int i = 0; i < instances.size(); i++) {
			double[] clusterInstance = features.row(instances.get(i));
			distance += this.getDistanceBetweenInstances(instance, clusterInstance);
		}
		return (distance / instances.size());
	}
	
	private double silhouetteCalcB(double[] instance, int cluster) {
		double smallestDist = Double.POSITIVE_INFINITY;
		for (int i = 0; i < this.numClusters; i++) {
			if (i != cluster) {
				double avgDistCluster = this.silhouetteCalcA(instance, i);
				if (avgDistCluster < smallestDist) {
					smallestDist = avgDistCluster;
				}
			}
		}
		return smallestDist;
	}
	
	private void printCentroids() {
		for (int i = 0; i < this.numClusters; i++) {
			System.out.println("\nCentroid " + i + ": ");
			ArrayList<Double> centroid = this.centroids.get(i);
			for (int j = 0; j < features.cols(); j++) {				
				if (centroid.get(j) == MISSING)
					System.out.print("?" + " ");
				else
					System.out.print(centroid.get(j) + " ");
			}
		}
		System.out.println();
	}
	
	private ArrayList<ArrayList<Double>> getRandomCentroids() {
		Random rand = new Random();
		ArrayList<Integer> randRows = new ArrayList<Integer>();
		while (randRows.size() < this.numClusters) {
			int randRow = rand.nextInt(this.features.rows());
			if (!randRows.contains(randRow)) {
				randRows.add(randRow);
			}
		}
		
		for (int i = 0; i < this.numClusters; i++) {
			int row = randRows.get(i);
			System.out.println("\nCentroid " + i + ": ");
			ArrayList<Double> centroid = this.centroids.get(i);
			for (int j = 0; j < features.cols(); j++) {
				centroid.add(features.get(row, j));
				
				if (features.get(row, j) == MISSING)
					System.out.print("?" + " ");
				else
					System.out.print(features.get(row, j) + " ");
			}
		}	
		return centroids;
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
	
	private double getDistanceBetweenInstances(double[] centroid, double[] instance) {
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
