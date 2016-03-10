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
	
	private void divideNode(DTNode node)
	{
		int numFeatures = node.getNumFeatures();
		for (int i = 0; i < numFeatures; i++)
		{
			if (!node.featureAlreadyUsed(i))
			{
				this.calcInfoGained(node, i);
			
				
			}
			
			
			
		}
		
		
		
	}
	
	private double calcInfoGained(DTNode node, int featureCol)
	{
		double info = node.calcInfo();
		int numFeatureValues = node.getNumFeatureValues(featureCol);
		for (int i = 0; i < numFeatureValues; i++)
		{
			
			
			
		}
		
		return 0;
	}
}
