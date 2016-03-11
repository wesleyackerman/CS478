package decisiontree;

import java.util.ArrayList;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner 
{
	private DTNode root;
	private DTNode bestTreeSoFar;
	private double bestAccSoFar;
	private Matrix validationFeatures;
	private Matrix validationLabels;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		features.replaceUnknowns();
		
		int numTrainingSet = (int)(features.rows() * 0.8);
		int numValidationSet = features.rows() - numTrainingSet;
		
		//Matrix trainingFeatures = new Matrix(features, 0, 0, numTrainingSet, features.cols());
		//Matrix trainingLabels = new Matrix(labels, 0, 0, numTrainingSet, labels.cols());
		//validationFeatures = new Matrix(features, numTrainingSet, 0,
		//		numValidationSet, features.cols());
		//validationLabels = new Matrix(labels, numTrainingSet, 0,
		//		numValidationSet, labels.cols());
		
		root = new DTNode();
		//root.setInstances(trainingFeatures, trainingLabels);
		root.setInstances(features, labels);
		root.incNodeCount();
		this.divideNode(root);
		//this.pruneRoot();
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		DTNode currentNode = this.root;
		while (currentNode.hasChildren())
		{
			int featureSplitOn = currentNode.getFeatureSplitOn();
			if (currentNode.getChild(features[featureSplitOn]) == null)
				break;
			currentNode = currentNode.getChild(features[featureSplitOn]);
		}
		double outputClass = currentNode.getLabel();
		labels[0] = outputClass;
	}
	
	private void divideNode(DTNode node) throws Exception
	{	
		//System.out.println("---------Dividing------------");
		int numFeatures = node.getNumFeatures();
		//double bestInfoGained = 0;
		double bestInfoRatio = 0;
		int featureToSplitOn = -1;
		
		for (int i = 0; i < numFeatures; i++)
		{
			if (!node.isFeatureUsed(i))
			{
				double infoGained = this.calcInfoGained(node, i);
				double splitInfo = node.calcSplitInfo(i);
				double infoRatio = splitInfo / infoGained;
				if (infoRatio > bestInfoRatio)
				{
					bestInfoRatio = infoRatio;
					featureToSplitOn = i;
				}
			}
		}
		if (featureToSplitOn == -1)
		{
			node.setLabel(node.getMostCommonLabel());
			//throw new Exception("Error. No features left to split on");
		}
		else
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
			node.incNodeCount();
			
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
			int instanceCount = node.getNumInstancesOfFeatureType(featureCol, i);
			Matrix instancesOfFeatureType = new Matrix(node.getInstances(), 0, 0, 0, node.getColumnCount());
			Matrix labelsOfFeatureType = new Matrix(node.getLabels(), 0, 0, 0, 1);
				
			node.getInstancesOfFeatureType(featureCol, i, instancesOfFeatureType, labelsOfFeatureType);
			DTNode tempNode = new DTNode(instancesOfFeatureType, labelsOfFeatureType);
			double tempNodeInfo = 0;
			if (!tempNode.isEmpty())
				tempNodeInfo = tempNode.calcInfo();
			infoLeft += ((double)instanceCount / (double)totalInstances) * tempNodeInfo;
		}
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
	
	private int getDepth(DTNode node)
	{
		if (node == null)
			return 0;
		//System.out.println("---------------------------");
		//System.out.println("Feature Split On " + node.getFeatureSplitOn());
		//if (node.getLabel() != null)
			//System.out.println("LABEL:" + node.getLabel());
		int numChildren = node.getChildCount();
		int deepestLevel = 0;
		if (numChildren != 0)
		{
			for (int i = 0; i < numChildren; i++)
			{	
				int childDepth = this.getDepth(node.getChild(i));
				if (deepestLevel < childDepth)
					deepestLevel = childDepth;
			}
			return deepestLevel+1;
		}
		else
			return 1;
	}
	
	private void pruneRoot() throws Exception
	{
		System.out.println("Depth: " + this.getDepth(root));
		System.out.println("Node count: " + root.getNodeCount());
		this.bestAccSoFar = this.measureAccuracy(validationFeatures, validationLabels, null);
		System.out.println("Accuracy: " + bestAccSoFar);
		bestTreeSoFar = new DTNode(root);
		this.pruneNode(root);
	}
	
	private void pruneNode(DTNode node) throws Exception
	{
		for (int i = 0; i < node.getChildCount(); i++)
		{
			this.pruneNode(node.getChild(i));
		}

		if (node.hasChildren())
		{
			int childCount = node.getChildCount();
			for (int i = 0; i < childCount; i++)
			{
				node.removeChild(i);
				node.decNodeCount();
			}
			node.setLabel(node.getMostCommonLabel());
			double acc = this.measureAccuracy(validationFeatures, validationLabels, null);
			System.out.println("Depth: " + this.getDepth(root));
			System.out.println("Node count: " + root.getNodeCount());
			System.out.println("Accuracy: " + acc);
			if (acc >= this.bestAccSoFar)
			{
				System.out.println("NEW Best Accuracy");
				this.bestAccSoFar = acc;
				this.bestTreeSoFar = new DTNode(root);
			}	
		}
	}
}
