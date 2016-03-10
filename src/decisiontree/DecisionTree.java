package decisiontree;

import java.util.ArrayList;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner 
{
	private DTNode root;
	private ArrayList<Integer> featuresUsed;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		root = new DTNode();
		featuresUsed = new ArrayList<Integer>();
		root.setInstances(features, labels);
		this.divideNode(root);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		DTNode currentNode = this.root;
		while (currentNode.hasChildren())
		{
			int featureSplitOn = currentNode.getFeatureSplitOn();
			currentNode = currentNode.getChild(features[featureSplitOn]);
		}
		double outputClass = currentNode.getLabel();
		labels[0] = outputClass;
	}
	
	private void divideNode(DTNode node) throws Exception
	{	
		int numFeatures = node.getNumFeatures();
		double bestInfoGained = 0;
		int featureToSplitOn = -1;
		for (int i = 0; i < numFeatures; i++)
		{
			if (!this.featuresUsed.contains(i))
			{
				double infoGained = this.calcInfoGained(node, i);
				if (infoGained > bestInfoGained)
				{
					bestInfoGained = infoGained;
					featureToSplitOn = i;
				}
			}
		}
		if (featureToSplitOn == -1)
			throw new Exception("Oops. No features left to split on");
		
		this.divideNodeOnFeature(node, featureToSplitOn);	
	}
	
	private void divideNodeOnFeature(DTNode node, int featureToSplitOn) throws Exception
	{
		this.featuresUsed.add(featureToSplitOn);
		node.setFeatureSplitOn(featureToSplitOn);
		int numFeatureValues = node.getNumFeatureValues(featureToSplitOn);
		for (int i = 0; i < numFeatureValues; i++)
		{
			Matrix childInstances = new Matrix();
			Matrix childLabels = new Matrix();
			node.getInstancesOfFeatureType(featureToSplitOn, i, childInstances, childLabels);
			
			DTNode child = new DTNode(childInstances, childLabels);
			node.addChild(child, i);
			
			if (this.isNodePartitionable(child, node))
				this.divideNode(child);
		}
	}
	
	private double calcInfoGained(DTNode node, int featureCol) throws Exception
	{
		double info = node.calcInfo();
		double infoLeft = 0;
		int totalInstances = node.getNumInstances();
		int numFeatureValues = node.getNumFeatureValues(featureCol);
		for (int i = 0; i < numFeatureValues; i++)
		{
			int instanceCount = node.getNumInstancesOfFeatureType(featureCol, i);
			Matrix instancesOfFeatureType = new Matrix();
			Matrix labelsOfFeatureType = new Matrix();
			node.getInstancesOfFeatureType(featureCol, i, instancesOfFeatureType, labelsOfFeatureType);
			DTNode tempNode = new DTNode(instancesOfFeatureType, labelsOfFeatureType);
			infoLeft += (instanceCount / totalInstances) * tempNode.calcInfo();	
		}
		return info - infoLeft;
	}
	
	private boolean isNodePartitionable(DTNode child, DTNode parent)
	{
		double nodePureLabel = child.isNodePure();
		double mostCommonLabel = parent.getMostCommonLabel();
		
		if (nodePureLabel != -1)
		{
			child.setLabel(nodePureLabel);
			return false;
		}
		else if (child.isEmpty())
		{
			child.setLabel(mostCommonLabel);
			return false;
		}
		else if (this.featuresUsed.size() == child.getNumFeatures())
		{
			child.setLabel(mostCommonLabel);
			return false;
		}
		return true;
	}
}
