package decisiontree;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner 
{
	private DTNode root;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		root = new DTNode();
		root.setInstances(features, labels);
		
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	

}
