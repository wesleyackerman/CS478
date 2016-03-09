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
	private ArrayList<String> attributesUsed;
	
	public DTNode()
	{
		this.label = null;
		this.instances = new Matrix();
		this.labels = new Matrix();
		this.children = new HashMap<Integer,DTNode>();
		this.attributesUsed = new ArrayList<String>();
	}

	public void setInstances(Matrix instances, Matrix labels)
	{
		this.instances = instances;
		this.labels = labels;
	}
}