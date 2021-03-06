package decisiontree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import toolkit.Matrix;

public class DTNode {
	private static int nodeCount = 0;
	private Double label;
	private Matrix instances;
	private Matrix labels;
	private Map<Integer, DTNode> children;
	private List<Integer> featuresUsed;
	private int featureSplitOn;
	
	public DTNode()
	{
		this.label = null;
		this.instances = null;
		this.labels = null;
		this.children = new HashMap<Integer,DTNode>();
		this.featureSplitOn = -1;
		this.featuresUsed = new ArrayList<Integer>();
	}
	
	public DTNode(Matrix instances, Matrix labels)
	{
		this.label = null;
		this.instances = instances;
		this.labels = labels;
		this.children = new HashMap<Integer,DTNode>();
		this.featureSplitOn = -1;
		this.featuresUsed = new ArrayList<Integer>();
	}
	
	public DTNode(DTNode that)
	{
		if (that == null)
			return;
		this.label = that.label;
		this.instances = that.instances;
		this.featureSplitOn = that.featureSplitOn;
		this.featuresUsed = that.featuresUsed;
		this.children = new HashMap<Integer,DTNode>();
		for (int i = 0; i < that.children.size(); i++)
		{
			DTNode child = new DTNode(that.children.get(i));
			this.children.put(i, child);
		}
	}

	public void setInstances(Matrix instances, Matrix labels)
	{
		this.instances = instances;
		this.labels = labels;
	}
	
	public int getNumFeatures()
	{
		return instances.cols();
	}
	
	public int getNumInstances()
	{
		return instances.rows();
	}
	
	public void setInstances(Matrix instances)
	{
		this.instances = instances;
	}
	
	public void setLabels(Matrix labels)
	{
		this.labels = labels;
	}
	
	public int getNumFeatureValues(int featureCol)
	{
		return this.instances.valueCount(featureCol);
	}
	
	public int getNumInstancesOfFeatureType(int featureCol, int featureType)
	{
		int count = 0;
		for (int j = 0; j < this.getNumInstances(); j++)
		{
			if (instances.get(j, featureCol) == featureType)
				count++;
		}
		return count;
	}
	
	public int getNumInstancesOfOutputType(int featureType)
	{
		int count = 0;
		for (int j = 0; j < this.getNumInstances(); j++)
		{
			if (labels.get(j, 0) == featureType)
				count++;
		}
		return count;
	}
	
	/*
	 * Returns the label if all instances associated with the node have the same label.
	 * Otherwise, returns a -1.
	 */
	public double isNodePure()
	{
		int numInstances = this.getNumInstances();
		double firstLabel = this.labels.get(0, 0);
		for (int i = 1; i < numInstances; i++)
		{
			if (this.labels.get(i, 0) != firstLabel)
				return -1;
		}
		return firstLabel;
	}
	
	public void getInstancesOfFeatureType(int featureCol, int featureType, 
			Matrix newInstances, Matrix newLabels) throws Exception
	{
		for (int i = 0; i < this.getNumInstances(); i++)
		{
			if (this.instances.get(i, featureCol) == featureType)
			{
				newInstances.add(instances, i, 0, 1);
				newLabels.add(labels, i, 0, 1);
			}
		}
	}

	public double calcInfo()
	{
		int totalInstances = instances.rows();
		int outputClassCount = labels.valueCount(0);
		double infoSum = 0;
		for (int i = 0; i < outputClassCount; i++)
		{
			int count = this.getNumInstancesOfOutputType(i);
			infoSum -= ((double)count / (double)totalInstances) * logBase2((double)count / (double)totalInstances);
		}
		return infoSum;
	}
	
	public double calcSplitInfo(int featureCol)
	{
		int totalInstances = instances.rows();
		int featureValueCount = this.getNumFeatureValues(featureCol);
		double infoSum = 0;
		for (int i = 0; i < featureValueCount; i++)
		{
			int count = this.getNumInstancesOfFeatureType(featureCol, i);
			infoSum -= ((double)count / (double)totalInstances) * logBase2((double)count / (double)totalInstances);
		}
		return infoSum;
	}
	
	public double logBase2(double x)
	{
		if (x == 0)
			return 0;
		return (Math.log(x) / Math.log(2));
	}
	
	public void setLabel(double label)
	{
		this.label = label;
	}
	
	public Double getLabel()
	{
		return this.label;
	}
	
	public void setFeatureSplitOn(int featureSplitOn)
	{
		this.featureSplitOn = featureSplitOn;
	}
	
	public int getFeatureSplitOn()
	{
		return this.featureSplitOn;
	}
	
	public boolean isEmpty()
	{
		if (instances.rows() == 0)
			return true;
		return false;
	}
	
	public DTNode getChild(double featureValue)
	{
		return this.children.get((int)featureValue);
	}
	
	public void addChild(DTNode child, int featureValue)
	{
		this.children.put(featureValue, child);
	}
	
	public double getMostCommonLabel()
	{
		return this.labels.mostCommonValue(0);
	}
	
	public boolean hasChildren()
	{
		return !this.children.isEmpty();
	}
	
	public int getChildCount()
	{
		return this.children.size();
	}
	
	public Matrix getInstances()
	{
		return this.instances;
	}
	
	public Matrix getLabels()
	{
		return this.labels;
	}
	
	public int getColumnCount()
	{
		return this.instances.cols();
	}
	
	public void addFeatureUsed(int feature)
	{
		this.featuresUsed.add(feature);
	}
	
	public List<Integer> getFeaturesUsed()
	{
		return this.featuresUsed;
	}
	
	public void setFeaturesUsed(List<Integer> featuresUsed)
	{
		this.featuresUsed = featuresUsed;
	}
	
	public boolean isFeatureUsed(int feature)
	{
		return this.featuresUsed.contains(feature);
	}
	
	public int getFeaturesUsedCount()
	{
		return this.featuresUsed.size();
	}
	
	public void incNodeCount()
	{
		nodeCount++;
	}
	
	public void decNodeCount()
	{
		nodeCount--;
	}
	
	public int getNodeCount()
	{
		return nodeCount;
	}
	
	public void removeChild(int key)
	{
		this.children.remove(key);
	}
}