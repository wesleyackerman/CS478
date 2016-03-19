package instancebasedlearner;

import java.util.ArrayList;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {
	private Matrix features;
	private Matrix labels;
	private final int kNearest = 3;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		this.features = features;
		this.labels = labels;	
	}
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		ArrayList<Integer> nearestNeighbors = new ArrayList<Integer>(kNearest);
		ArrayList<Double> nnDistances = new ArrayList<Double>(kNearest);
		// Set each of the NN distances to infinity, since there are no NN yet
		for (int i = 0; i < kNearest; i++)
			nnDistances.set(i, Double.POSITIVE_INFINITY);
		
		for (int i = 0; i < this.features.rows(); i++) {
			double distance = this.calcDistance(features, this.features.row(i));
			
			
		}	
	}
	
	private double calcDistance(double[] instance1, double[] instance2) {
		features.
		
		
		
		return -1;
	}

}
