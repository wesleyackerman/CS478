package decisiontree;

import java.util.ArrayList;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner 
{
	private DTNode root;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		features.replaceUnknowns();
		root = new DTNode();
		root.setInstances(features, labels);
		this.divideNode(root);
		//this.printNode(root, 0);
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
		System.out.println("---------Dividing------------");
		int numFeatures = node.getNumFeatures();
		double bestInfoGained = 0;
		int featureToSplitOn = -1;
		
		for (int i = 0; i < numFeatures; i++)
		{
			if (!node.isFeatureUsed(i))
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
		{
			System.out.println(node.calcInfo());
			System.out.println("hi");
			throw new Exception("Oops. No features left to split on");
		}
		
		this.divideNodeOnFeature(node, featureToSplitOn);	
	}
	
	private void divideNodeOnFeature(DTNode node, int featureToSplitOn) throws Exception
	{
		//System.out.println("~~~Splitting on feature: " + featureToSplitOn);
		node.addFeatureUsed(featureToSplitOn);
		node.setFeatureSplitOn(featureToSplitOn);
		int numFeatureValues = node.getNumFeatureValues(featureToSplitOn);
		for (int i = 0; i < numFeatureValues; i++)
		{
			Matrix childInstances = new Matrix(node.getInstances(), 0, 0, 0, node.getColumnCount());
			Matrix childLabels = new Matrix(node.getLabels(), 0, 0, 0, 1);
			node.getInstancesOfFeatureType(featureToSplitOn, i, childInstances, childLabels);
			
			DTNode child = new DTNode(childInstances, childLabels);
			child.setFeaturesUsed(new ArrayList<Integer>(node.getFeaturesUsed()));
			node.addChild(child, i);
			
			if (this.isNodePartitionable(child, node))
			{	
				this.divideNode(child);
			}
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
			if (!node.isFeatureUsed(i))
			{
				int instanceCount = node.getNumInstancesOfFeatureType(featureCol, i);
				Matrix instancesOfFeatureType = new Matrix(node.getInstances(), 0, 0, 0, node.getColumnCount());
				Matrix labelsOfFeatureType = new Matrix(node.getLabels(), 0, 0, 0, 1);
				
				node.getInstancesOfFeatureType(featureCol, i, instancesOfFeatureType, labelsOfFeatureType);
				DTNode tempNode = new DTNode(instancesOfFeatureType, labelsOfFeatureType);
				infoLeft += ((double)instanceCount / (double)totalInstances) * tempNode.calcInfo();
			}
		}
		System.out.println("Info left: " + infoLeft);
		return info - infoLeft;
	}
	
	private boolean isNodePartitionable(DTNode child, DTNode parent)
	{
		double mostCommonLabel = parent.getMostCommonLabel();
		if (child.isEmpty())
		{
			child.setLabel(mostCommonLabel);
			return false;
		}
		
		double nodePureLabel = child.isNodePure();
		if (nodePureLabel != -1)
		{
			child.setLabel(nodePureLabel);
			return false;
		}
		else if (parent.getFeaturesUsedCount() == parent.getNumFeatures())
		{
			child.setLabel(mostCommonLabel);
			return false;
		}
		return true;
	}
	
	private void printNode(DTNode node, int level)
	{
		System.out.println("---------------------------");
		System.out.println("Node level: " + level);
		System.out.println("Feature Split On " + node.getFeatureSplitOn());
		if (node.getLabel() != null)
			System.out.println("LABEL:" + node.getLabel());
		int numChildren = node.getChildCount();
		System.out.println("CHILD count: " + numChildren);
		for (int i = 0; i < numChildren; i++)
		{	
			printNode(node.getChild(i), ++level);
		}
	}
}
