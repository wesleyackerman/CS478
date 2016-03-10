package decisiontree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import toolkit.Matrix;

public class DTNode {
	private String label;
	private Matrix instances;
	private Matrix labels;
	private Map<Integer, DTNode> children;
	private ArrayList<Integer> featuresAlreadyUsed;
	
	public DTNode()
	{
		this.label = null;
		this.instances = new Matrix();
		this.labels = new Matrix();
		this.children = new HashMap<Integer,DTNode>();
		this.featuresAlreadyUsed = new ArrayList<Integer>();
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
	
	public boolean featureAlreadyUsed(int featureCol)
	{
		return this.featuresAlreadyUsed.contains(featureCol);
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
		Map<Integer,Integer> instanceCountPerClass = 
				new HashMap<Integer,Integer>(outputClassCount);
		double infoSum = 0;
		for (int i = 0; i < outputClassCount; i++)
		{
			int count = this.getNumInstancesOfOutputType(i);
			instanceCountPerClass.put(i, count);
			infoSum -= (count / totalInstances) * logBase2(count / totalInstances);
		}
		return infoSum;
	}
	
	public double logBase2(double x)
	{
		return (Math.log(x)/Math.log(2));
	}
}